"""Does the transferable residual survive when induced errors are removed?

CaptainCook4D's deviations are largely experimenter-assigned: participants were
told to follow a recipe or to induce errors from designed categories. If the
"off-protocol residual" is mostly those assigned perturbations, its 8/8 fold
positivity is an artifact of the experimental design, not evidence of tacit
practice that transfers between operators.

This re-runs the leave-one-participant-out headroom analysis on slices of the
data, holding the metric (MRR), the split (LOPO over 8 participants), the seed,
and the rankers identical to the original run.

TWO BUGS ARE FIXED FIRST, in every slice including the reproduced baseline.
Both corrupt the step ORDER, which is the only thing the residual model reads:

  fix 1  287 steps carry start_time == -1: they were skipped and never
         performed, yet they occupy a position in the step array. 141 recordings
         are affected, 85 with the phantom interior. Transitions into and out of
         them never happened.
  fix 2  in 72 recordings the array order is not chronological (83 adjacent
         inversions). Array order was being taken as execution order.

SLICES (rule B, chosen because it never splices a sequence):

  as_published   no fixes, all recordings -- reproduces the published number
  all            fixes applied, all recordings -- the comparable baseline
  nonerror       fixes + is_error == False  (164 recordings)  PRIMARY
  error_only     fixes + is_error == True   (220 recordings)  contrast
  sensitivity_C  fixes + recordings carrying no Order/Missing error

Rule B excludes whole recordings, never individual steps. Dropping an interior
error step would splice its neighbours and fabricate a (prev -> next) adjacency
that never occurred, and the residual model is a bigram over exactly those
adjacencies -- so step-level exclusion would manufacture the signal being
measured.
"""

from __future__ import annotations

import collections
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import DATA_DIR, Execution, _norm_recipe, build_decisions, load_protocols
from headroom import SEED, ResidualModel, rank_protocol, rank_protocol_residual, reciprocal_rank

PILOT_RECIPES = ["blenderbananapancakes", "broccolistirfry", "buttercorncup",
                 "capresebruschetta", "cheesepimiento"]
ORDERING_TAGS = {"Order Error", "Missing Step"}


def _error_index():
    """recording_id -> (is_error, set of error tags anywhere in the recording)."""
    raw = json.load(open(os.path.join(DATA_DIR, "annotation_json", "error_annotations.json")))
    out = {}
    for r in raw:
        tags = set()
        for s in r["step_annotations"]:
            for e in s.get("errors") or []:
                tags.add(e["tag"])
        out[r["recording_id"]] = (bool(r["is_error"]), tags)
    return out


def load_executions(protocols, apply_fixes: bool):
    """Recordings as ordered step sequences, optionally with the two order fixes."""
    raw = json.load(open(os.path.join(DATA_DIR, "annotation_json",
                                      "complete_step_annotations.json")))
    out, stats = [], collections.Counter()
    for rec in raw.values():
        recipe = _norm_recipe(rec["activity_name"])
        proto = protocols.get(recipe)
        if proto is None:
            continue
        steps = [s for s in rec["steps"] if s["step_id"] in proto.steps]
        if apply_fixes:
            n0 = len(steps)
            steps = [s for s in steps if s["start_time"] != -1]      # fix 1
            stats["phantom_steps_dropped"] += n0 - len(steps)
            order = [s["start_time"] for s in steps]
            if order != sorted(order):
                stats["recordings_reordered"] += 1
            steps = sorted(steps, key=lambda s: s["start_time"])     # fix 2
        if len(steps) < 2:
            stats["recordings_dropped_too_short"] += 1
            continue
        out.append(Execution(
            recording_id=rec["recording_id"], recipe=recipe,
            person_id=rec["person_id"], environment=rec["environment"],
            step_ids=[s["step_id"] for s in steps],
            durations={s["step_id"]: 0.0 for s in steps},
            has_errors={s["step_id"]: bool(s.get("has_errors")) for s in steps},
        ))
    return sorted(out, key=lambda e: e.recording_id), stats


SLICES = {
    "as_published":  dict(fixes=False, keep=lambda rid, isr, tags: True),
    "all":           dict(fixes=True,  keep=lambda rid, isr, tags: True),
    "nonerror":      dict(fixes=True,  keep=lambda rid, isr, tags: not isr),
    "error_only":    dict(fixes=True,  keep=lambda rid, isr, tags: isr),
    "sensitivity_C": dict(fixes=True,  keep=lambda rid, isr, tags: not (ORDERING_TAGS & tags)),
}


def run_slice(name, protocols, errors, recipes=None):
    spec = SLICES[name]
    execs, fixstats = load_executions(protocols, spec["fixes"])
    execs = [e for e in execs if spec["keep"](e.recording_id, *errors[e.recording_id])]
    if recipes is not None:
        execs = [e for e in execs if e.recipe in recipes]

    decisions = []
    for ex in execs:
        decisions.extend(build_decisions(ex, protocols[ex.recipe]))

    persons = sorted({e.person_id for e in execs})
    by_person = collections.defaultdict(list)
    for d in decisions:
        by_person[d.person_id].append(d)

    fold = {}
    for held in persons:
        train = [e for e in execs if e.person_id != held]
        evald = by_person[held]
        if not train or not evald:
            continue
        # strict operator hold-out, asserted in both directions
        assert all(e.person_id != held for e in train), "operator leak in authoring set"
        assert all(d.person_id == held for d in evald), "operator leak in eval set"

        model = ResidualModel(train)
        rng = random.Random(SEED + held)
        prot = statistics.mean(reciprocal_rank(rank_protocol(d, rng), d.true_next) for d in evald)
        rng = random.Random(SEED + held)
        resid = statistics.mean(
            reciprocal_rank(rank_protocol_residual(d, rng, model), d.true_next) for d in evald)
        fold[held] = {"protocol": prot, "residual": resid, "delta": resid - prot,
                      "n_decisions": len(evald)}

    deltas = [f["delta"] for f in fold.values()]
    return {
        "slice": name,
        "fixes_applied": spec["fixes"],
        "n_recordings": len(execs),
        "n_decisions": len(decisions),
        "n_folds": len(fold),
        "mrr_protocol_only": statistics.mean(f["protocol"] for f in fold.values()),
        "mrr_protocol_plus_residual": statistics.mean(f["residual"] for f in fold.values()),
        "headroom_mean": statistics.mean(deltas),
        "folds_positive": sum(d > 0 for d in deltas),
        "per_fold": {str(k): v for k, v in sorted(fold.items())},
        "fix_stats": dict(fixstats),
    }


def report(results, stage):
    L = []
    L.append("=" * 92)
    L.append(f"NON-ERROR HEADROOM  ({stage})   metric MRR | LOPO over participants | seed {SEED}")
    L.append("=" * 92)
    L.append(f"{'slice':<16}{'recs':>6}{'decisions':>11}{'protocol':>10}{'+residual':>11}"
             f"{'headroom':>11}{'signs':>8}")
    for name in ["as_published", "all", "nonerror", "error_only", "sensitivity_C"]:
        r = results[name]
        L.append(f"{name:<16}{r['n_recordings']:>6}{r['n_decisions']:>11}"
                 f"{r['mrr_protocol_only']:>10.4f}{r['mrr_protocol_plus_residual']:>11.4f}"
                 f"{r['headroom_mean']:>+11.4f}{r['folds_positive']:>5}/{r['n_folds']}")
    L.append("")
    L.append("PER-PARTICIPANT HEADROOM (residual minus protocol-only)")
    hdr = "participant".ljust(16) + "".join(f"{p:>9}" for p in
                                            results["all"]["per_fold"])
    L.append(hdr)
    for name in ["as_published", "all", "nonerror", "error_only", "sensitivity_C"]:
        r = results[name]
        row = name.ljust(16)
        for p in results["all"]["per_fold"]:
            v = r["per_fold"].get(p)
            row += f"{v['delta']:>+9.4f}" if v else f"{'--':>9}"
        L.append(row)
    return "\n".join(L)


def main(stage="pilot"):
    protocols = load_protocols()
    errors = _error_index()
    recipes = PILOT_RECIPES if stage == "pilot" else None

    results = {n: run_slice(n, protocols, errors, recipes) for n in SLICES}
    text = report(results, stage)
    print(text)

    os.makedirs("results", exist_ok=True)
    json.dump({"stage": stage, "seed": SEED, "metric": "MRR",
               "split": "leave-one-participant-out over 8 participants",
               "rule": "B -- whole-recording exclusion; no step-level splicing",
               "slices": results},
              open(f"results/error_slice_{stage}.json", "w"), indent=2)
    open(f"results/error_slice_{stage}.txt", "w").write(text + "\n")
    print(f"\nwrote results/error_slice_{stage}.json")
    return results


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pilot")
