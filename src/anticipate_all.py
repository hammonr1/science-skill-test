"""Stage 3: all 8 leave-one-participant-out folds, covered points only.

Same metric, arms, seeds, library and leakage boundary as the fold-6 dry run in
anticipate.py. This adds what a single fold cannot give:

  per-fold, per-arm Brier
  fold-level SD of each planned contrast
  MDE at n=8 computed FROM THESE OBSERVED SDs, not carried over from the MRR
    analysis, which used a different metric on a different scoring set
  fold-win counts for the contrasts that matter (C vs D, C vs E)
  an idiosyncrasy check: whether a contrast is carried by one participant

The result is still labelled underpowered. Coverage is 17% of decision points and
each fold contributes 10-40 positives, so a contrast is being estimated on a few
hundred paired points spread across 8 folds whose libraries differ (Jaccard
0.536). The MDE below is the honest statement of what that buys.
"""

from __future__ import annotations

import collections
import json
import math
import os
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anthropic

from anticipate import (ARMS, ARM_LABEL, MAX_TOKENS, MODEL, SEED, auroc, brier,
                        assert_no_target_leakage, build_prompt, call,
                        covered_points, parse_prob, render_block)
from data import load_protocols
from evaluate import Cache
from labels import build_points
from library_churn import load_records

CONTRASTS = [("C", "D"), ("C", "E"), ("C", "A"), ("B", "A"), ("E", "A")]


def jobs_for_fold(fold, protocols, records, points, err_lookup):
    pts, lib = covered_points(fold, protocols, records, points)
    stocked = sorted(rc for rc, rs in lib.items() if rs)
    donor_of = {rc: stocked[(i + 1) % len(stocked)] for i, rc in enumerate(stocked)}
    jobs = []
    for p in pts:
        proto = protocols[p.recipe]
        base = None
        for arm in ARMS:
            if arm == "E":
                dr = donor_of.get(p.recipe, p.recipe)
                block = render_block(lib[dr], "E", protocols[dr])
            else:
                block = render_block(lib[p.recipe], arm, proto)
            u = build_prompt(p, proto, block)
            assert_no_target_leakage(u, p, err_lookup,
                                     proto.steps.get(p.target_step_id, ""))
            cut = u.index("PRACTITIONER TIPS") if "PRACTITIONER TIPS" in u \
                else u.index("THE STEP THEY ARE ABOUT")
            residue = u[:cut].rstrip() + "\n" + u[u.index("THE STEP THEY ARE ABOUT"):]
            if base is None:
                base = residue
            assert residue == base, f"arms differ outside skill block at {p.recording_id}#{p.index}"
            jobs.append({"fold": fold, "arm": arm, "point": p, "user": u,
                         "n_block_tokens": len(block.split())})
    return jobs, len(pts), sum(p.y_execution for p in pts)


def mde(sd, n=8, alpha=0.05, power=0.80):
    """Two-sided alpha .05, power .80 -> (1.96 + 0.84) = 2.80."""
    return 2.80 * sd / math.sqrt(n)


def main(workers=12):
    protocols = load_protocols()
    _, records = load_records()
    points = build_points()

    err_lookup = {}
    ea = {r["recording_id"]: r for r in json.load(open(
        "data/annotation_json/error_annotations.json"))}
    for p in points:
        for a in ea[p.recording_id]["step_annotations"]:
            if a["step_id"] == p.target_step_id and abs(a["start_time"] - p.target_start_time) < 1e-6:
                toks = [e.get("description") for e in (a.get("errors") or [])]
                toks.append(a.get("modified_description"))
                err_lookup[(p.recording_id, p.index)] = [t for t in toks if t]
                break

    folds = sorted({p.person_id for p in points})
    all_jobs, meta = [], {}
    for f in folds:
        j, n, pos = jobs_for_fold(f, protocols, records, points, err_lookup)
        all_jobs += j
        meta[f] = {"n_points": n, "n_positives": pos,
                   "base_rate": pos / n if n else float("nan")}
    print(f"{len(all_jobs)} calls over {len(folds)} folds "
          f"({sum(m['n_points'] for m in meta.values())} covered points x {len(ARMS)} arms)")

    client = anthropic.Anthropic()
    cache = Cache()
    done = [0]; lock = threading.Lock()

    def work(j):
        t, cached = call(client, cache, j["user"])
        with lock:
            done[0] += 1
            if done[0] % 400 == 0:
                print(f"  {done[0]}/{len(all_jobs)}", flush=True)
        return {"fold": j["fold"], "arm": j["arm"], "y": j["point"].y_execution,
                "p": parse_prob(t), "recording_id": j["point"].recording_id,
                "index": j["point"].index, "n_block_tokens": j["n_block_tokens"],
                "cached": cached, "raw": (t or "").strip()[:24]}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(work, all_jobs))

    unpars = collections.Counter(r["arm"] for r in rows if r["p"] is None)

    # per fold, per arm
    per = collections.defaultdict(dict)
    for f in folds:
        for arm in ARMS:
            rs = [r for r in rows if r["fold"] == f and r["arm"] == arm and r["p"] is not None]
            ps = [r["p"] for r in rs]; ys = [r["y"] for r in rs]
            per[f][arm] = {"n": len(rs), "brier": brier(ps, ys), "auroc": auroc(ps, ys),
                           "p_mean": statistics.mean(ps)}

    pooled = {}
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm and r["p"] is not None]
        ps = [r["p"] for r in rs]; ys = [r["y"] for r in rs]
        pooled[arm] = {"label": ARM_LABEL[arm], "n": len(rs),
                       "brier": brier(ps, ys), "auroc": auroc(ps, ys),
                       "p_mean": statistics.mean(ps),
                       "block_tokens_mean": statistics.mean(
                           r["n_block_tokens"] for r in rows if r["arm"] == arm)}

    contrasts = {}
    for x, y in CONTRASTS:
        d = [per[f][x]["brier"] - per[f][y]["brier"] for f in folds]
        m = statistics.mean(d); sd = statistics.stdev(d)
        # leave-one-fold-out: is the contrast carried by a single participant?
        loo = {str(folds[i]): statistics.mean(d[:i] + d[i+1:]) for i in range(len(d))}
        worst = min(loo.items(), key=lambda kv: abs(kv[1]))
        contrasts[f"{x}-{y}"] = {
            "per_fold": {str(f): v for f, v in zip(folds, d)},
            "mean": m, "sd": sd, "se": sd / math.sqrt(len(d)),
            "ci95": [m - 1.96 * sd / math.sqrt(len(d)), m + 1.96 * sd / math.sqrt(len(d))],
            "mde_n8": mde(sd), "detectable": abs(m) > mde(sd),
            "folds_x_better": sum(1 for v in d if v < 0),   # lower Brier is better
            "loo_means": loo,
            "sign_flips_if_fold_dropped": [k for k, v in loo.items() if (v < 0) != (m < 0)],
            "most_influential_fold": worst[0],
        }

    L = ["=" * 92,
         "STAGE 3 -- all 8 folds, covered points only.  UNDERPOWERED; see MDE.",
         "=" * 92,
         f"  model {MODEL}  temp 0  seed {SEED}  max_tokens {MAX_TOKENS}",
         f"  metric Brier (primary, lower better), AUROC (secondary)",
         f"  library: replacement-pair deltas, no support floor",
         "",
         f"  {'fold':<6}{'points':>8}{'pos':>6}{'base':>8}" + "".join(f"{a:>9}" for a in ARMS)]
    for f in folds:
        L.append(f"  {f:<6}{meta[f]['n_points']:>8}{meta[f]['n_positives']:>6}"
                 f"{meta[f]['base_rate']:>8.3f}" +
                 "".join(f"{per[f][a]['brier']:>9.4f}" for a in ARMS))
    L += ["",
          f"  {'POOLED':<6}{sum(m['n_points'] for m in meta.values()):>8}"
          f"{sum(m['n_positives'] for m in meta.values()):>6}{'':>8}" +
          "".join(f"{pooled[a]['brier']:>9.4f}" for a in ARMS),
          "",
          f"{'arm':<5}{'description':<28}{'blk tok':>9}{'Brier':>9}{'AUROC':>8}{'p_mean':>9}"]
    for a in ARMS:
        q = pooled[a]
        L.append(f"{a:<5}{q['label']:<28}{q['block_tokens_mean']:>9.1f}"
                 f"{q['brier']:>9.4f}{q['auroc']:>8.3f}{q['p_mean']:>9.3f}")
    L += ["",
          f"{'contrast':<10}{'mean':>10}{'fold SD':>10}{'95% CI':>22}{'MDE(n=8)':>11}"
          f"{'detect?':>9}{'folds':>7}"]
    for k, c in contrasts.items():
        L.append(f"{k:<10}{c['mean']:>+10.4f}{c['sd']:>10.4f}"
                 f"{'[%+.4f,%+.4f]' % tuple(c['ci95']):>22}{c['mde_n8']:>11.4f}"
                 f"{('YES' if c['detectable'] else 'no'):>9}{c['folds_x_better']:>4}/8")
    L += ["", "IDIOSYNCRASY (leave-one-fold-out mean; sign flips = carried by one participant)"]
    for k, c in contrasts.items():
        fl = c["sign_flips_if_fold_dropped"]
        L.append(f"  {k:<10} most influential fold {c['most_influential_fold']:>2}"
                 f"   sign flips when dropped: {fl if fl else 'none'}")
    if unpars:
        L.append(f"\n  WARNING unparseable by arm: {dict(unpars)}")
    text = "\n".join(L)
    print("\n" + text)

    os.makedirs("results", exist_ok=True)
    json.dump({"stage": "stage3_all_folds", "model": MODEL, "seed": SEED,
               "max_tokens": MAX_TOKENS, "underpowered": True,
               "scoring": "covered points only (cue+target match), no support floor",
               "fold_meta": {str(k): v for k, v in meta.items()},
               "per_fold": {str(f): per[f] for f in folds},
               "pooled": pooled, "contrasts": contrasts,
               "n_calls": len(all_jobs),
               "n_cached": sum(1 for r in rows if r["cached"]),
               "unparseable_by_arm": dict(unpars)},
              open("results/anticipate_all.json", "w"), indent=2)
    json.dump(rows, open("results/anticipate_all_rows.json", "w"), indent=2)
    open("results/anticipate_all.txt", "w").write(text + "\n")
    print("\nwrote results/anticipate_all.json")


if __name__ == "__main__":
    main()
