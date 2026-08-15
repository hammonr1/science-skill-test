"""Diagnostic: how much does the diff-based library churn across folds?

The bigram library is near-constant across the 8 leave-one-participant-out folds
(2 distinct libraries; fold 1 differs by one rule). That means its fold-level
variance is almost entirely evaluation-side, so the CI on a contrast reflects
which decisions were scored, not which skills were authored.

An execution-template library is mined from a far sparser source -- the
description -> modified_description deltas on execution-error steps -- so
dropping one participant may materially change it. If it does, fold variance
absorbs authoring instability and the MDE inflates. That is a power risk worth
knowing BEFORE spending on model calls, which is why this runs offline.

RULE CONSTRUCTION (deterministic; model-normalised clustering is deferred).
  cue          the step most often completed immediately before the target,
               among training participants -- observable strictly before the
               target step begins
  target       the step the intervention applies to; step index is unchanged
  intervention a replacement pair, 'protocol span -> actual span', aligned
               between the protocol description and what the operator actually
               did (modified_description), taken over execution-error instances
               only, most frequent delta wins

Normalisation preserves numeric atoms and is otherwise case, whitespace and
punctuation; the delta is a difflib alignment rather than a set difference. Both
are deterministic -- no model, no lexicon, no hand-labelling.
"""

from __future__ import annotations

import collections
import difflib
import itertools
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import DATA_DIR, _norm_recipe, load_protocols
from labels import EXECUTION_TAGS
from skills import mine_residual

K = 8


def norm(s: str) -> str:
    """Deterministic normalisation: case, whitespace, punctuation.

    Numeric atoms are preserved. Blanket punctuation stripping split '1/2' into
    '1 2', so the delta between '1 tablespoon' and '1/2 tablespoon' came out as
    the bare token '2' -- the quantity change, which IS the intervention, was
    destroyed by normalisation. Fractions, decimals and ranges are therefore
    matched as single tokens before punctuation is removed from anything else.
    Still deterministic: no model, no lexicon, no hand-labelling.
    """
    s = s.lower()
    toks = re.findall(r"\d+\s*/\s*\d+|\d+\s*-\s*\d+|\d+\.\d+|\d+|[a-z]+", s)
    return " ".join(t.replace(" ", "") for t in toks)


CONTEXT = 1   # equal tokens kept either side of a changed span


def delta(protocol_text: str, actual_text: str) -> str:
    """Replacement-pair delta: 'protocol span -> actual span', with context.

    An additions-only delta drops whatever matched on both sides, which is
    usually the unit: '1 tablespoon' -> '1/2 tablespoon' reduced to the bare
    token '1/2'. Aligning the two token sequences and emitting the CHANGED SPAN
    with one equal token either side keeps the referent, so the same pair reads
    '1 tablespoon -> 1/2 tablespoon'.

    Deterministic: difflib alignment, no model, no lexicon.
    """
    p, a = norm(protocol_text).split(), norm(actual_text).split()
    sm = difflib.SequenceMatcher(a=p, b=a, autojunk=False)
    ops = sm.get_opcodes()
    parts = []
    for k, (tag, i1, i2, j1, j2) in enumerate(ops):
        if tag == "equal":
            continue
        lo = ops[k - 1] if k else None
        hi = ops[k + 1] if k + 1 < len(ops) else None
        pre = p[max(lo[2] - CONTEXT, lo[1]):lo[2]] if lo and lo[0] == "equal" else []
        post = p[hi[1]:min(hi[1] + CONTEXT, hi[2])] if hi and hi[0] == "equal" else []
        old = " ".join(pre + p[i1:i2] + post).strip()
        new = " ".join(pre + a[j1:j2] + post).strip()
        if old and new and old != new:
            parts.append(f"{old} -> {new}")
        elif new:
            parts.append(f"add {new}")
        elif old:
            parts.append(f"omit {old}")
    return "; ".join(parts)


def load_records():
    """(recipe, person, recording, ordered performed steps with tags and texts)."""
    protocols = load_protocols()
    comp = json.load(open(os.path.join(DATA_DIR, "annotation_json",
                                       "complete_step_annotations.json")))
    ea = {r["recording_id"]: r for r in
          json.load(open(os.path.join(DATA_DIR, "annotation_json",
                                      "error_annotations.json")))}
    out = []
    for rec in comp.values():
        recipe = _norm_recipe(rec["activity_name"])
        proto = protocols.get(recipe)
        if proto is None:
            continue
        paired = list(zip(rec["steps"], ea[rec["recording_id"]]["step_annotations"]))
        paired = [(a, b) for a, b in paired
                  if a["step_id"] in proto.steps and a["start_time"] != -1]
        paired.sort(key=lambda ab: ab[0]["start_time"])
        if len(paired) < 2:
            continue
        out.append({"recipe": recipe, "person": rec["person_id"],
                    "recording_id": rec["recording_id"], "paired": paired})
    return protocols, out


def mine_execution_library(records, recipe, k=K):
    """Top-k (cue, target, intervention) rules for one recipe from given records."""
    interventions = collections.defaultdict(collections.Counter)   # target -> delta
    predecessors = collections.defaultdict(collections.Counter)    # target -> prev step
    for r in records:
        if r["recipe"] != recipe:
            continue
        paired = r["paired"]
        for i, (a, b) in enumerate(paired):
            tags = {e["tag"] for e in (b.get("errors") or [])}
            if not (tags & EXECUTION_TAGS):
                continue
            mod = b.get("modified_description")
            if not mod:
                continue
            d = delta(a["description"], mod)
            if not d:
                continue
            interventions[a["step_id"]][d] += 1
            if i > 0:
                predecessors[a["step_id"]][paired[i - 1][0]["step_id"]] += 1

    rules = []
    for target, c in interventions.items():
        d, n = c.most_common(1)[0]
        cue = predecessors[target].most_common(1)[0][0] if predecessors[target] else None
        rules.append({"recipe": recipe, "cue_step": cue, "target_step": target,
                      "intervention": d, "support": n})
    rules.sort(key=lambda r: (-r["support"], r["target_step"]))
    return rules[:k]


def rule_key(r):
    return (r["recipe"], r["cue_step"], r["target_step"], r["intervention"])


def churn(libs):
    """Pairwise Jaccard, singleton count, and distinct-library count."""
    keys = {f: {rule_key(r) for r in rs} for f, rs in libs.items()}
    folds = sorted(keys)
    jac = []
    for x, y in itertools.combinations(folds, 2):
        a, b = keys[x], keys[y]
        jac.append(len(a & b) / len(a | b) if (a | b) else 1.0)
    counts = collections.Counter(k for s in keys.values() for k in s)
    return {
        "n_pairs": len(jac),
        "jaccard_mean": statistics.mean(jac) if jac else float("nan"),
        "jaccard_min": min(jac) if jac else float("nan"),
        "jaccard_max": max(jac) if jac else float("nan"),
        "distinct_libraries": len({frozenset(v) for v in keys.values()}),
        "union_rules": len(counts),
        "rules_in_all_8": sum(1 for v in counts.values() if v == 8),
        "rules_unique_to_one_fold": sum(1 for v in counts.values() if v == 1),
        "per_fold_size": {str(f): len(keys[f]) for f in folds},
    }


def main():
    protocols, records = load_records()
    persons = sorted({r["person"] for r in records})
    recipes = sorted(protocols)

    # ---- diff-based execution library, per fold
    exec_libs = {}
    supports = []
    for held in persons:
        train = [r for r in records if r["person"] != held]
        assert all(r["person"] != held for r in train), "operator leak in authoring set"
        lib = []
        for rc in recipes:
            rs = mine_execution_library(train, rc)
            lib += rs
            supports += [r["support"] for r in rs]
        exec_libs[held] = lib

    # ---- bigram library, per fold, same folds, for the baseline comparison
    from data import load_all
    _, execs, _ = load_all()
    bg_libs = {}
    bg_support = []
    for held in persons:
        tr = [e for e in execs if e.person_id != held]
        lib = []
        for rc in recipes:
            for s in mine_residual(tr, protocols, rc, k=K):
                lib.append({"recipe": rc, "cue_step": s.cue_step,
                            "target_step": s.action_step, "intervention": "",
                            "support": s.support})
                bg_support.append(s.support)
        bg_libs[held] = lib

    ec, bc = churn(exec_libs), churn(bg_libs)

    def fmt(name, c, sup):
        L = [f"--- {name}",
             f"    rules per fold            {c['per_fold_size']}",
             f"    union of rules            {c['union_rules']}",
             f"    distinct libraries / 8    {c['distinct_libraries']}",
             f"    pairwise Jaccard (28 pr)  mean {c['jaccard_mean']:.4f}"
             f"  min {c['jaccard_min']:.4f}  max {c['jaccard_max']:.4f}",
             f"    rules present in all 8    {c['rules_in_all_8']}",
             f"    rules unique to one fold  {c['rules_unique_to_one_fold']}"]
        if sup:
            q = statistics.quantiles(sup, n=4) if len(sup) > 3 else [float('nan')]*3
            L += [f"    support: n {len(sup)}  mean {statistics.mean(sup):.2f}"
                  f"  median {statistics.median(sup):.1f}  min {min(sup)}  max {max(sup)}",
                  f"             quartiles {[round(x,1) for x in q]}",
                  f"             support==1 rules: {sum(1 for x in sup if x==1)}"
                  f" ({100*sum(1 for x in sup if x==1)/len(sup):.0f}%)"]
        return "\n".join(L)

    text = "\n\n".join([
        "=" * 78,
        "LIBRARY CHURN DIAGNOSTIC (offline, no model calls)",
        "=" * 78,
        fmt("DIFF-BASED EXECUTION LIBRARY", ec, supports),
        fmt("BIGRAM LIBRARY (baseline)", bc, bg_support),
    ])
    print(text)

    os.makedirs("results", exist_ok=True)
    json.dump({"K": K, "execution_library": ec, "bigram_library": bc,
               "execution_support": {"n": len(supports),
                                     "mean": statistics.mean(supports) if supports else None,
                                     "median": statistics.median(supports) if supports else None,
                                     "min": min(supports) if supports else None,
                                     "max": max(supports) if supports else None,
                                     "n_support_1": sum(1 for x in supports if x == 1)}},
              open("results/library_churn.json", "w"), indent=2)
    open("results/library_churn.txt", "w").write(text + "\n")
    print("\nwrote results/library_churn.json")


if __name__ == "__main__":
    main()
