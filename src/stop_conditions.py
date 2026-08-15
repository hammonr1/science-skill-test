"""Pre-registered stop conditions for the pilot.

Any trip means the arm contrast is contaminated and the full run would be wasted
money. Checked mechanically from the pilot rows so the decision is not a judgement
call made after seeing the effect.

  1. n_malformed differs by more than 2x between any two arms
  2. mean_omitted_per_response differs by more than 1.0 between any two arms
  3. any arm returns >5% malformed responses
  4. assert_prompts_differ_only_in_skill_block fails on any job
  5. actual spend exceeds $10

Condition 5 carries a measurement caveat: call_model records only response text,
not the usage block, so realised token spend cannot be recovered from this run.
The estimate is reported instead and the gap is stated rather than papered over.
"""

from __future__ import annotations

import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import omission_diagnostics

SPEND_CAP_USD = 10.0
HAIKU_IN, HAIKU_OUT = 1.00, 5.00   # $/Mtok, claude-haiku-4-5


def check(rows, cost_estimate, prompt_assertion_passed, chars_per_token=3.6):
    om = omission_diagnostics(rows)
    arms = sorted(om)
    trips, notes = [], []

    # 1. malformed ratio between arms
    mal = {a: om[a]["n_malformed"] for a in arms}
    worst = None
    for x, y in itertools.combinations(arms, 2):
        a, b = mal[x], mal[y]
        hi, lo = max(a, b), min(a, b)
        ratio = float("inf") if (lo == 0 and hi > 0) else (hi / lo if lo else 1.0)
        if worst is None or ratio > worst[0]:
            worst = (ratio, x, y, a, b)
        if ratio > 2.0:
            trips.append(f"COND 1: n_malformed {x}={a} vs {y}={b} (ratio {ratio:.2f} > 2)")
    notes.append(f"cond1 worst malformed ratio: {worst[1]}={worst[3]} vs {worst[2]}={worst[4]}"
                 f" -> {worst[0]:.2f}")

    # 2. mean omitted per response spread
    mo = {a: om[a]["mean_omitted_per_response"] for a in arms}
    spread = max(mo.values()) - min(mo.values())
    if spread > 1.0:
        trips.append(f"COND 2: mean_omitted_per_response spread {spread:.4f} > 1.0  {mo}")
    notes.append(f"cond2 mean_omitted spread: {spread:.4f}")

    # 3. malformed rate per arm
    for a in arms:
        frac = om[a]["frac_malformed"]
        if frac > 0.05:
            trips.append(f"COND 3: arm {a} malformed rate {frac:.4%} > 5%")
    notes.append("cond3 malformed rates: "
                 + ", ".join(f"{a}={om[a]['frac_malformed']:.4%}" for a in arms))

    # 4. prompt identity
    if not prompt_assertion_passed:
        trips.append("COND 4: assert_prompts_differ_only_in_skill_block FAILED")
    notes.append(f"cond4 prompt-identity assertion passed: {prompt_assertion_passed}")

    # 5. spend
    est_in = cost_estimate["est_input_tokens"] * (4.0 / chars_per_token)
    est_out = cost_estimate["est_output_tokens"]
    est_usd = est_in / 1e6 * HAIKU_IN + est_out / 1e6 * HAIKU_OUT
    if est_usd > SPEND_CAP_USD:
        trips.append(f"COND 5: estimated spend ${est_usd:.2f} > ${SPEND_CAP_USD:.2f}")
    notes.append(f"cond5 estimated spend: ${est_usd:.2f} (NOT measured -- usage blocks "
                 f"are not recorded by call_model; this is a chars/{chars_per_token} estimate)")

    return {"tripped": trips, "notes": notes, "omission": om,
            "estimated_spend_usd": round(est_usd, 2),
            "spend_is_measured": False}


def main(stage="pilot"):
    rows = json.load(open(f"results/{stage}_rows.json"))
    meta = json.load(open(f"results/{stage}_results.json"))
    res = check(rows, meta["cost_estimate"],
                prompt_assertion_passed=bool(meta.get("prompt_identity_decisions_checked")))
    print("=" * 74)
    print("PRE-REGISTERED STOP CONDITIONS")
    print("=" * 74)
    for n in res["notes"]:
        print("  " + n)
    print()
    if res["tripped"]:
        print("  TRIPPED:")
        for t in res["tripped"]:
            print("    - " + t)
    else:
        print("  none tripped")
    json.dump(res, open(f"results/{stage}_stop_conditions.json", "w"), indent=2)
    print(f"\nwrote results/{stage}_stop_conditions.json")
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pilot")
