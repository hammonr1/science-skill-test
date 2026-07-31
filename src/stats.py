"""Aggregation, confidence intervals, and multiple-comparison correction.

Two levels of uncertainty are reported, and they answer different questions:

  fold-level CI  -- bootstrap over the 8 held-out participants. This is the one
                    that matters for the claim "transfers to an UNSEEN operator",
                    because participants are the unit of generalisation. With 8
                    folds it is necessarily wide; that is honest, not a defect.

  paired test    -- the arm contrasts (C-B, C-D, B-A) are computed PAIRED on the
                    same decision points, which removes decision difficulty as a
                    nuisance term and is far more sensitive than comparing two
                    marginal means.

Multiple comparisons: three planned contrasts, Holm-Bonferroni corrected.
"""

from __future__ import annotations

import collections
import math
import random
import statistics

CONTRASTS = [("C", "B"), ("C", "D"), ("B", "A")]


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=11):
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.mean(s))
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return (lo, hi)


def cluster_bootstrap_paired(rows, arm_x, arm_y, n_boot=10000, seed=13):
    """Paired contrast, resampling PARTICIPANTS (the unit of generalisation).

    Returns (mean difference, CI, two-sided bootstrap p-value).
    """
    by_fold = collections.defaultdict(dict)
    for r in rows:
        by_fold[r["fold"]].setdefault(r["arm"], {})[(r["recording_id"], r["index"])] = r["rr"]

    fold_deltas = {}
    for fold, arms in by_fold.items():
        if arm_x not in arms or arm_y not in arms:
            continue
        keys = set(arms[arm_x]) & set(arms[arm_y])
        if keys:
            fold_deltas[fold] = statistics.mean(arms[arm_x][k] - arms[arm_y][k] for k in keys)

    folds = sorted(fold_deltas)
    vals = [fold_deltas[f] for f in folds]
    if len(vals) < 2:
        return float("nan"), (float("nan"), float("nan")), float("nan"), fold_deltas

    obs = statistics.mean(vals)
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        s = [vals[rng.randrange(len(vals))] for _ in vals]
        boot.append(statistics.mean(s))
    boot.sort()
    lo, hi = boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot)]
    # two-sided p: proportion of centred bootstrap means at least as extreme as 0
    centred = [b - obs for b in boot]
    p = sum(1 for c in centred if abs(c) >= abs(obs)) / n_boot
    return obs, (lo, hi), p, fold_deltas


def holm(pvals: dict[str, float]) -> dict[str, tuple[float, bool]]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = max(prev, min(1.0, (m - i) * p))
        prev = adj
        out[k] = (adj, adj < 0.05)
    return out


def summarize(rows):
    by_arm = collections.defaultdict(list)
    fold_means = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by_arm[r["arm"]].append(r)
        fold_means[r["arm"]][r["fold"]].append(r["rr"])

    table = {}
    for arm, rs in by_arm.items():
        fm = [statistics.mean(v) for v in fold_means[arm].values()]
        table[arm] = {
            "n": len(rs),
            "mrr": statistics.mean(x["rr"] for x in rs),
            "top1": statistics.mean(x["top1"] for x in rs),
            "fold_means": fm,
            "ci": bootstrap_ci(fm),
        }

    contrasts, pvals = {}, {}
    for x, y in CONTRASTS:
        if x in by_arm and y in by_arm:
            d, ci, p, per_fold = cluster_bootstrap_paired(rows, x, y)
            contrasts[f"{x}-{y}"] = {"delta": d, "ci": ci, "p_raw": p,
                                     "per_fold": per_fold}
            pvals[f"{x}-{y}"] = p
    for k, (adj, sig) in holm(pvals).items():
        contrasts[k]["p_holm"] = adj
        contrasts[k]["significant_holm"] = sig
    return table, contrasts


def format_report(table, contrasts, arm_label, ceiling=None):
    L = []
    L.append("=" * 78)
    L.append("RESULTS  (MRR of true next action; leave-one-participant-out)")
    L.append("=" * 78)
    L.append(f"{'arm':<6}{'description':<26}{'MRR':>8}{'95% CI (participants)':>26}{'top-1':>9}")
    for arm in ["A", "B", "C", "D"]:
        if arm not in table:
            continue
        t = table[arm]
        L.append(f"{arm:<6}{arm_label[arm]:<26}{t['mrr']:>8.4f}"
                 f"{'[%.4f, %.4f]' % t['ci']:>26}{t['top1']:>9.4f}")
    if ceiling is not None:
        L.append(f"{'--':<6}{'ceiling (stat. residual)':<26}{ceiling:>8.4f}{'':>26}{'':>9}")
    L.append("")
    L.append("PLANNED CONTRASTS (paired on identical decision points,")
    L.append("bootstrap over participants, Holm-corrected across 3 tests)")
    L.append(f"{'contrast':<12}{'delta MRR':>11}{'95% CI':>24}{'p_raw':>9}{'p_holm':>9}{'sig':>6}")
    for k, c in contrasts.items():
        L.append(f"{k:<12}{c['delta']:>+11.4f}{'[%+.4f, %+.4f]' % c['ci']:>24}"
                 f"{c['p_raw']:>9.4f}{c.get('p_holm', float('nan')):>9.4f}"
                 f"{'YES' if c.get('significant_holm') else 'no':>6}")
    return "\n".join(L)
