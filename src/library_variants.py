"""Three support floors for the diff-based execution library, side by side.

Floors compared:
  none          every mined (cue, target, intervention) rule
  min2_inst     >= 2 INSTANCES of the intervention
  min2_ops      >= 2 DISTINCT OPERATORS contributing the intervention

Support is always computed on that fold's TRAINING executions only. Computing it
globally would let the held-out participant decide whether a rule survives, which
is operator leakage into library formation even though the rule text itself never
came from the held-out traces.

Coverage is the binding constraint and is reported two ways:
  coverage_target      fraction of the fold's decision points whose target step
                       has any rule in that fold's library
  coverage_cue_target  fraction where a rule matches BOTH the just-completed step
                       (the cue, observable strictly before the target) and the
                       target step -- this is the template actually being tested
"""

from __future__ import annotations

import collections
import itertools
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import load_protocols
from labels import EXECUTION_TAGS, build_points
from library_churn import delta, load_records

K = 8
VARIANTS = ["none", "min2_inst", "min2_ops"]


def mine(records, recipe, variant, k=K):
    """Top-k rules for one recipe. `records` are TRAINING records only."""
    inst = collections.defaultdict(collections.Counter)          # target -> delta -> n
    ops = collections.defaultdict(lambda: collections.defaultdict(set))  # target -> delta -> {person}
    preds = collections.defaultdict(collections.Counter)         # target -> prev step

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
            inst[a["step_id"]][d] += 1
            ops[a["step_id"]][d].add(r["person"])
            if i > 0:
                preds[a["step_id"]][paired[i - 1][0]["step_id"]] += 1

    rules = []
    for target, c in inst.items():
        for d, n in c.items():
            n_ops = len(ops[target][d])
            if variant == "min2_inst" and n < 2:
                continue
            if variant == "min2_ops" and n_ops < 2:
                continue
            cue = preds[target].most_common(1)[0][0] if preds[target] else None
            rules.append({"recipe": recipe, "cue_step": cue, "target_step": target,
                          "intervention": d, "support": n, "n_operators": n_ops})
    # one rule per target: the best-supported intervention, then top-k by support
    best = {}
    for r in sorted(rules, key=lambda r: (-r["n_operators"], -r["support"])):
        best.setdefault(r["target_step"], r)
    out = sorted(best.values(), key=lambda r: (-r["support"], r["target_step"]))
    return out[:k]


def key(r):
    return (r["recipe"], r["cue_step"], r["target_step"], r["intervention"])


def coverage(lib, points):
    """Fraction of this fold's decision points a library rule speaks to."""
    by_target = collections.defaultdict(list)
    for r in lib:
        by_target[(r["recipe"], r["target_step"])].append(r)
    n = t_hit = ct_hit = 0
    for p in points:
        n += 1
        rs = by_target.get((p.recipe, p.target_step_id))
        if not rs:
            continue
        t_hit += 1
        cue = p.done[-1] if p.done else None
        if any(r["cue_step"] == cue for r in rs):
            ct_hit += 1
    return (t_hit / n if n else 0.0), (ct_hit / n if n else 0.0), n


def run():
    protocols = load_protocols()
    _, records = load_records()
    points = build_points()
    recipes = sorted(protocols)
    persons = sorted({r["person"] for r in records})
    pts_by_fold = collections.defaultdict(list)
    for p in points:
        pts_by_fold[p.person_id].append(p)

    out = {}
    for variant in VARIANTS:
        libs, supports, cov_t, cov_ct = {}, [], {}, {}
        for held in persons:
            train = [r for r in records if r["person"] != held]
            assert all(r["person"] != held for r in train), "operator leak in authoring set"
            lib = []
            for rc in recipes:
                rs = mine(train, rc, variant)
                lib += rs
                supports += [r["support"] for r in rs]
            libs[held] = lib
            ct, cct, _ = coverage(lib, pts_by_fold[held])
            cov_t[held], cov_ct[held] = ct, cct

        keys = {f: {key(r) for r in rs} for f, rs in libs.items()}
        jac = [len(keys[x] & keys[y]) / len(keys[x] | keys[y])
               for x, y in itertools.combinations(persons, 2) if keys[x] | keys[y]]
        counts = collections.Counter(k for s in keys.values() for k in s)
        out[variant] = {
            "rules_per_fold": {str(f): len(keys[f]) for f in persons},
            "rules_per_fold_mean": statistics.mean(len(keys[f]) for f in persons),
            "union_rules": len(counts),
            "jaccard_mean": statistics.mean(jac) if jac else float("nan"),
            "jaccard_min": min(jac) if jac else float("nan"),
            "jaccard_max": max(jac) if jac else float("nan"),
            "rules_in_all_8": sum(1 for v in counts.values() if v == 8),
            "rules_unique_to_one_fold": sum(1 for v in counts.values() if v == 1),
            "singleton_fraction": (sum(1 for v in counts.values() if v == 1) / len(counts)
                                   if counts else float("nan")),
            "support_mean": statistics.mean(supports) if supports else float("nan"),
            "support_median": statistics.median(supports) if supports else float("nan"),
            "support_eq_1_frac": (sum(1 for x in supports if x == 1) / len(supports)
                                  if supports else float("nan")),
            "coverage_target": {str(f): cov_t[f] for f in persons},
            "coverage_cue_target": {str(f): cov_ct[f] for f in persons},
            "coverage_target_mean": statistics.mean(cov_t.values()),
            "coverage_cue_target_mean": statistics.mean(cov_ct.values()),
        }

    L = ["=" * 96,
         "SUPPORT-FLOOR VARIANTS  (per-fold support, training executions only)",
         "=" * 96,
         f"{'variant':<12}{'rules/fold':>11}{'union':>7}{'Jaccard':>9}{'in all 8':>10}"
         f"{'uniq-1':>8}{'sup=1':>8}{'cov_tgt':>9}{'cov_cue+tgt':>13}"]
    for v in VARIANTS:
        d = out[v]
        L.append(f"{v:<12}{d['rules_per_fold_mean']:>11.1f}{d['union_rules']:>7}"
                 f"{d['jaccard_mean']:>9.3f}{d['rules_in_all_8']:>10}"
                 f"{d['rules_unique_to_one_fold']:>8}{d['support_eq_1_frac']:>8.0%}"
                 f"{d['coverage_target_mean']:>9.1%}{d['coverage_cue_target_mean']:>13.1%}")
    L.append("")
    L.append("PER-FOLD COVERAGE (cue+target match -- the template under test)")
    L.append(f"{'variant':<12}" + "".join(f"{p:>8}" for p in persons))
    for v in VARIANTS:
        L.append(f"{v:<12}" + "".join(f"{out[v]['coverage_cue_target'][str(p)]:>8.1%}"
                                      for p in persons))
    L.append("")
    L.append("PER-FOLD JACCARD RANGE")
    for v in VARIANTS:
        d = out[v]
        L.append(f"  {v:<12} mean {d['jaccard_mean']:.3f}  min {d['jaccard_min']:.3f}"
                 f"  max {d['jaccard_max']:.3f}")
    text = "\n".join(L)
    print(text)

    os.makedirs("results", exist_ok=True)
    json.dump({"K": K, "support_scope": "per fold, training executions only",
               "variants": out}, open("results/library_variants.json", "w"), indent=2)
    open("results/library_variants.txt", "w").write(text + "\n")
    print("\nwrote results/library_variants.json")
    return out


if __name__ == "__main__":
    run()
