"""Emit results/final.json -- the quotable record of the four-arm test.

Gate 6 deliverable. For each planned contrast: point estimate, 95% CI, per-fold
values, the number of PAIRED decision points the contrast actually rests on, and
both raw and Holm-adjusted p. Plus the per-arm parse diagnostics from Gate 3,
because differential omission across arms is a confound and belongs next to the
numbers it could distort.

The paired n is deliberately recomputed here rather than inherited from the arm
totals: a contrast is only defined on decisions where BOTH arms produced a
ranking, and quoting the arm-level n would overstate the evidence.
"""

from __future__ import annotations

import collections
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import ANNOTATIONS_COMMIT
from evaluate import ARM_LABEL, ARMS, RANK_SEED, omission_diagnostics
from prompts import SYSTEM, USER_TEMPLATE
from stats import CONTRASTS, bootstrap_ci, cluster_bootstrap_paired, holm

N_BOOT = 10000
ALPHA = 0.05


def _leakage():
    """Ship the cue-leakage numbers inside final.json.

    The conditional lift is the same magnitude as the entire oracle skill effect,
    so anyone quoting a C-vs-B gap out of this file must see it in the same file.
    """
    p = "results/diagnostics.json"
    if not os.path.exists(p):
        return None
    leak = json.load(open(p))["leakage"]
    leak["interpretation"] = (
        "marginal = lexical attack on cue text vs random; conditional = same attack "
        "on top of the task graph every arm already sees. The conditional value is "
        "the decision-relevant one, and it is comparable in size to the oracle K=8 "
        "skill effect (+0.0173), so a C-vs-B gap of that order is not safely "
        "attributable to cue conditioning."
    )
    return leak


def paired_n(rows, arm_x, arm_y):
    """Decision points where both arms produced a ranking, per fold and total."""
    by_fold = collections.defaultdict(dict)
    for r in rows:
        by_fold[r["fold"]].setdefault(r["arm"], set()).add((r["recording_id"], r["index"]))
    per_fold, total = {}, 0
    for fold, arms in by_fold.items():
        if arm_x in arms and arm_y in arms:
            k = len(arms[arm_x] & arms[arm_y])
            per_fold[str(fold)] = k
            total += k
    return total, per_fold


def build(rows, stage, model, k_skills, seed):
    arms_present = [a for a in ARMS if any(r["arm"] == a for r in rows)]

    by_arm = collections.defaultdict(list)
    fold_means = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by_arm[r["arm"]].append(r)
        fold_means[r["arm"]][r["fold"]].append(r["rr"])

    per_arm = {}
    for arm in arms_present:
        rs = by_arm[arm]
        fm = {str(f): statistics.mean(v) for f, v in sorted(fold_means[arm].items())}
        # MICRO: every decision point weighted equally; CI resamples POINTS.
        # MACRO: every participant weighted equally; CI resamples PARTICIPANTS.
        # They must not be mixed. The previous version paired a micro point
        # estimate with a macro CI, so the estimate was not the centre of its own
        # interval and a reader deriving C-A from the arm column got a number
        # that did not match the paired macro contrast.
        micro = statistics.mean(x["rr"] for x in rs)
        macro = statistics.mean(fm.values())
        per_arm[arm] = {
            "label": ARM_LABEL[arm],
            "n_decision_points": len(rs),
            "mrr_micro": micro,
            "mrr_micro_ci95_point_bootstrap": list(
                bootstrap_ci([x["rr"] for x in rs], n_boot=N_BOOT, alpha=ALPHA)),
            "mrr_macro": macro,
            "mrr_macro_ci95_participant_bootstrap": list(
                bootstrap_ci(list(fm.values()), n_boot=N_BOOT, alpha=ALPHA)),
            "top1_micro": statistics.mean(x["top1"] for x in rs),
            "top1_macro": statistics.mean(
                statistics.mean(1.0 if r["rank"] == 1 else 0.0 for r in by_arm[arm]
                                if r["fold"] == f) for f in sorted(fold_means[arm])),
            "per_fold_mrr": fm,
        }

    contrasts, pvals = {}, {}
    for x, y in CONTRASTS:
        if x not in by_arm or y not in by_arm:
            continue
        delta, ci, p, per_fold = cluster_bootstrap_paired(rows, x, y, n_boot=N_BOOT)
        n_total, n_per_fold = paired_n(rows, x, y)
        # micro version of the same contrast: paired per decision point,
        # bootstrap resampling POINTS rather than participants
        xa = {(r["recording_id"], r["index"]): r["rr"] for r in rows if r["arm"] == x}
        ya = {(r["recording_id"], r["index"]): r["rr"] for r in rows if r["arm"] == y}
        keys = sorted(set(xa) & set(ya))
        diffs = [xa[k] - ya[k] for k in keys]
        micro_delta = statistics.mean(diffs) if diffs else float("nan")
        micro_ci = list(bootstrap_ci(diffs, n_boot=N_BOOT, alpha=ALPHA))
        contrasts[f"{x}-{y}"] = {
            "arms": [x, y],
            "description": f"{ARM_LABEL[x]}  minus  {ARM_LABEL[y]}",
            "delta_macro": delta,
            "ci95_macro_participant_bootstrap": list(ci),
            "delta_micro": micro_delta,
            "ci95_micro_point_bootstrap": micro_ci,
            "point_estimate_delta_mrr": delta,   # retained: macro, the inferential one
            "ci95": list(ci),
            "per_fold_delta": {str(k): v for k, v in sorted(per_fold.items())},
            "n_folds": len(per_fold),
            "n_paired_decision_points": n_total,
            "n_paired_decision_points_per_fold": n_per_fold,
            "p_raw": p,
        }
        pvals[f"{x}-{y}"] = p

    for key, (adj, sig) in holm(pvals).items():
        contrasts[key]["p_holm"] = adj
        contrasts[key]["significant_holm_alpha05"] = sig

    return {
        "stage": stage,
        "model": model,
        "annotations_commit": ANNOTATIONS_COMMIT,
        "seed": seed,
        "rank_fallback_seed": RANK_SEED,
        "k_skills": k_skills,
        "metric": "MRR of the true next step among all remaining steps",
        "aggregation_note": (
            "MICRO = mean over decision points, CI resamples decision points. "
            "MACRO = mean over the 8 participant folds, CI resamples participants. "
            "The MACRO figures are the inferential ones: participants are the unit "
            "of generalisation, and fold sizes here range 64-198, so micro lets a "
            "large fold dominate. Quote one aggregation throughout and say which. "
            "point_estimate_delta_mrr and ci95 are the MACRO values, kept under "
            "their original names so earlier citations remain valid."),
        "holdout": "leave-one-participant-out; skills authored from training participants only",
        "bootstrap": {
            "type": "paired cluster bootstrap, resampling participants",
            "n_resamples": N_BOOT,
            "alpha": ALPHA,
            "correction": "Holm-Bonferroni across the 3 planned contrasts",
        },
        "per_arm": per_arm,
        "contrasts": contrasts,
        "omission_diagnostics": omission_diagnostics(rows),
        "cue_leakage_diagnostic": _leakage(),
        "prompt_templates": {"system": SYSTEM, "user": USER_TEMPLATE},
    }


def main(stage="pilot"):
    rows = json.load(open(f"results/{stage}_rows.json"))
    meta = json.load(open(f"results/{stage}_results.json"))
    out = build(rows, stage, meta["model"], meta["k_skills"], meta["seed"])
    json.dump(out, open("results/final.json", "w"), indent=2)

    c = out["contrasts"]
    print("=" * 78)
    print(f"FINAL  stage={stage}  model={out['model']}")
    print("=" * 78)
    print(f"{'arm':<5}{'description':<26}{'MRR micro':>11}{'95% CI (pts)':>22}"
          f"{'MRR macro':>11}{'95% CI (ppl)':>22}")
    for arm, a in out["per_arm"].items():
        cim = "[%.4f, %.4f]" % tuple(a["mrr_micro_ci95_point_bootstrap"])
        cia = "[%.4f, %.4f]" % tuple(a["mrr_macro_ci95_participant_bootstrap"])
        print(f"{arm:<5}{a['label']:<26}{a['mrr_micro']:>11.4f}{cim:>22}"
              f"{a['mrr_macro']:>11.4f}{cia:>22}")
    print()
    print(f"{'contrast':<10}{'MACRO':>10}{'95% CI (ppl)':>24}{'MICRO':>10}"
          f"{'95% CI (pts)':>24}{'p_holm':>9}{'n':>7}")
    for k, v in c.items():
        cia = "[%+.4f, %+.4f]" % tuple(v["ci95_macro_participant_bootstrap"])
        cim = "[%+.4f, %+.4f]" % tuple(v["ci95_micro_point_bootstrap"])
        print(f"{k:<10}{v['delta_macro']:>+10.4f}{cia:>24}{v['delta_micro']:>+10.4f}"
              f"{cim:>24}{v['p_holm']:>9.4f}{v['n_paired_decision_points']:>7}")
    if "C-D" in c:
        v = c["C-D"]
        print()
        print(f"C-D (load-bearing) MACRO {v['delta_macro']:+.4f} "
              f"[{v['ci95_macro_participant_bootstrap'][0]:+.4f}, "
              f"{v['ci95_macro_participant_bootstrap'][1]:+.4f}]  |  MICRO "
              f"{v['delta_micro']:+.4f} "
              f"[{v['ci95_micro_point_bootstrap'][0]:+.4f}, "
              f"{v['ci95_micro_point_bootstrap'][1]:+.4f}]")
        print(f"  p_holm={v['p_holm']:.4f}, {v['n_folds']} folds, "
              f"{v['n_paired_decision_points']} paired points. Quote MACRO.")
    print("\nwrote results/final.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pilot")
