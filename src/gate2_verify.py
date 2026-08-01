"""Gate 2 verification: is the protocol/execution join real and non-vacuous?

Three checks, all assertions rather than prose:
  1. every performed step_id resolves to non-empty protocol text
  2. the precedence relation actually bites (some remaining step is illegal)
  3. count steps that fail to project, and characterise the decisions they touch
"""

from __future__ import annotations

import collections
import json
import sys

from data import (_activity_global_ids, _norm_recipe, load_all,
                  load_protocols, load_step_descriptions, DATA_DIR)
import os


def unprojected_report():
    """Steps present in a recording but absent from the projected protocol.

    Recomputed from the RAW annotations, before load_executions filters, so the
    filter's effect is visible rather than hidden.
    """
    protocols = load_protocols()
    raw = json.load(open(os.path.join(DATA_DIR, "annotation_json",
                                      "complete_step_annotations.json")))
    n_steps = n_bad = 0
    bad_by_recipe = collections.Counter()
    recs_touched = set()
    bad_ids = collections.Counter()
    for rec in raw.values():
        recipe = _norm_recipe(rec["activity_name"])
        proto = protocols.get(recipe)
        if proto is None:
            continue
        for s in rec["steps"]:
            n_steps += 1
            if s["step_id"] not in proto.steps:
                n_bad += 1
                bad_by_recipe[recipe] += 1
                bad_ids[(recipe, s["step_id"], s["description"][:60])] += 1
                recs_touched.add(rec["recording_id"])
    return {
        "performed_steps_total": n_steps,
        "unprojected_steps": n_bad,
        "unprojected_by_recipe": dict(bad_by_recipe),
        "recordings_touched": sorted(recs_touched),
        "distinct_unprojected": [list(k) + [v] for k, v in bad_ids.items()],
    }


def main():
    protocols, executions, decisions = load_all()
    out = {}

    # ---- check 1: every performed step has non-empty protocol text
    blanks = []
    for e in executions:
        proto = protocols[e.recipe]
        for sid in e.step_ids:
            txt = proto.steps.get(sid, "")
            if not txt.strip():
                blanks.append((e.recording_id, sid))
    assert not blanks, f"steps with empty protocol text: {blanks[:10]}"
    out["check1_all_steps_have_text"] = True
    out["performed_step_instances"] = sum(len(e.step_ids) for e in executions)

    # ---- check 2: precedence is non-vacuous
    n_bite = 0
    illegal_counts = []
    for d in decisions:
        n_illegal = len(d.candidates) - len(d.ready)
        illegal_counts.append(n_illegal)
        if n_illegal > 0:
            n_bite += 1
    frac = n_bite / len(decisions)
    assert frac > 0.5, f"precedence looks vacuous: only {frac:.4f} of decisions constrained"
    out["check2_frac_decisions_with_an_illegal_remaining_step"] = frac
    out["mean_illegal_remaining_per_decision"] = sum(illegal_counts) / len(illegal_counts)
    out["mean_candidates"] = sum(len(d.candidates) for d in decisions) / len(decisions)
    out["mean_ready"] = sum(len(d.ready) for d in decisions) / len(decisions)
    out["n_decisions"] = len(decisions)

    # ---- check 3: projection failures
    out["check3_projection"] = unprojected_report()

    print("=" * 74)
    print("GATE 2 VERIFICATION")
    print("=" * 74)
    print(f"decisions                                  : {out['n_decisions']}")
    print(f"performed step instances                   : {out['performed_step_instances']}")
    print(f"check 1  all steps resolve to text         : PASS")
    print(f"mean candidates / mean ready               : "
          f"{out['mean_candidates']:.2f} / {out['mean_ready']:.2f}")
    print(f"check 2  decisions with an ILLEGAL option  : "
          f"{frac:.4f}  ({n_bite}/{len(decisions)})")
    print(f"         mean illegal remaining per decision: "
          f"{out['mean_illegal_remaining_per_decision']:.2f}")
    p = out["check3_projection"]
    print(f"check 3  unprojected performed steps       : "
          f"{p['unprojected_steps']} / {p['performed_steps_total']}")
    if p["unprojected_steps"]:
        print(f"         by recipe                         : {p['unprojected_by_recipe']}")
        print(f"         recordings touched                : {len(p['recordings_touched'])}")
        for row in p["distinct_unprojected"]:
            print(f"         - {row}")

    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/gate2_verify.json", "w"), indent=2)
    print("\nwrote results/gate2_verify.json")
    return out


if __name__ == "__main__":
    main()
