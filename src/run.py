"""Single reproducible entrypoint.

  python src/run.py --stage precheck   # dataset + power + leakage, no API
  python src/run.py --stage pilot      # 5 recipes, all 4 arms
  python src/run.py --stage full       # all 24 recipes
  python src/run.py --stage pilot --dry-run   # cost estimate only
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import ANNOTATIONS_COMMIT, load_all, load_executions
from evaluate import (ARM_LABEL, RANK_SEED, assert_prompts_differ_only_in_skill_block,
                      build_jobs, estimate_cost, omission_diagnostics, run_jobs)
from prompts import SYSTEM, USER_TEMPLATE
from stats import format_report, summarize

SEED = 7
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
# fixed alphabetically so the pilot subset is not cherry-picked
PILOT_RECIPES = ["blenderbananapancakes", "broccolistirfry", "buttercorncup",
                 "capresebruschetta", "cheesepimiento"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["precheck", "pilot", "full"], default="precheck")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--k-skills", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    protocols, executions, decisions = load_all()

    print(f"CaptainCook4D annotations @ {ANNOTATIONS_COMMIT}")
    print(f"recipes {len(protocols)} | executions {len(executions)} | "
          f"participants {len(set(e.person_id for e in executions))} | decisions {len(decisions)}")

    if args.stage == "precheck":
        import diagnostics
        import headroom
        import power
        headroom.run()
        print()
        diagnostics.run()
        print()
        power.run()
        return

    recipes = PILOT_RECIPES if args.stage == "pilot" else None
    jobs = build_jobs(protocols, executions, decisions, recipes=recipes,
                      k_skills=args.k_skills, seed=SEED)

    # Gate 6 hygiene, asserted in code rather than inspected: the four arms must
    # be byte-identical outside the skill block, and one model serves all arms.
    n_checked = assert_prompts_differ_only_in_skill_block(jobs)
    print(f"prompt-identity assertion passed over {n_checked} decisions x 4 arms")
    assert len({j["arm"] for j in jobs}) == 4, "expected exactly four arms"

    est = estimate_cost(jobs, args.model)
    print(f"\nstage={args.stage} model={args.model} k_skills={args.k_skills}")
    print(f"cost estimate: {est['calls']} calls | ~{est['est_input_tokens']:,} in / "
          f"~{est['est_output_tokens']:,} out tokens | ~${est['est_usd']}")
    if args.dry_run:
        json.dump(est, open(f"results/{args.stage}_cost_estimate.json", "w"), indent=2)
        return

    rows = run_jobs(jobs, args.model, workers=args.workers)
    table, contrasts = summarize(rows)
    omissions = omission_diagnostics(rows)

    print("\nPARSE HEALTH (differential omission across arms is a confound)")
    print(f"{'arm':<5}{'responses':>11}{'with omission':>15}{'mean omitted':>14}"
          f"{'malformed':>11}")
    for arm, o in omissions.items():
        print(f"{arm:<5}{o['n_responses']:>11}{o['n_responses_with_omission']:>15}"
              f"{o['mean_omitted_per_response']:>14.4f}{o['n_malformed']:>11}")

    hp = "results/headroom.json"
    ceiling = None
    if os.path.exists(hp):
        ceiling = statistics.mean(json.load(open(hp))["mrr"]["3_protocol_plus_residual"])

    report = format_report(table, contrasts, ARM_LABEL, ceiling=ceiling)
    print("\n" + report)

    json.dump(
        {
            "stage": args.stage, "model": args.model, "seed": SEED,
            "rank_fallback_seed": RANK_SEED,
            "k_skills": args.k_skills, "annotations_commit": ANNOTATIONS_COMMIT,
            "recordings_dropped_unprojected": len(getattr(load_executions, "dropped", [])),
            "prompt_identity_decisions_checked": n_checked,
            "system_prompt": SYSTEM, "user_template": USER_TEMPLATE,
            "cost_estimate": est, "omission_diagnostics": omissions,
            "table": {k: {kk: vv for kk, vv in v.items()} for k, v in table.items()},
            "contrasts": contrasts, "ceiling_statistical": ceiling,
        },
        open(f"results/{args.stage}_results.json", "w"), indent=2, default=str)
    open(f"results/{args.stage}_report.txt", "w").write(report)
    json.dump(rows, open(f"results/{args.stage}_rows.json", "w"), indent=2)
    print(f"\nwrote results/{args.stage}_results.json")


if __name__ == "__main__":
    main()
