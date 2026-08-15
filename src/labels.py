"""Stage 1: per-decision-point binary labels for the execution regime.

New scored task: given context strictly prior to a step instance, predict whether
that instance is executed with an EXECUTION-QUALITY error.

RULING APPLIED (user, Stage 0 gate):
  positive  = target step carries one or more of
              {Preparation, Measurement, Timing, Technique, Temperature} Error
  negative  = everything else, including all Order Error points and all
              Missing Step points, which stay in the denominator
  'Other'   = negative for all 8 instances, no hand-labelling
  the 4 Missing-Step-on-performed-steps = negative by tag rule, no per-instance
              discretion

POSITIONAL LABELLING (hard rule 2). Labels are read by ARRAY INDEX after zipping
complete_step_annotations against error_annotations, never by step_id lookup.
The two files agree on (step_id, start_time) at all 5700 array positions, so the
zip is exact. step_id lookup is unsafe: 13 recordings repeat a step_id with
differing flags, and a step_id can appear both as a phantom and as a performed
instance -- that ambiguity is what made an earlier count report 7 surviving
Missing Step tags where the true figure under the pilot filter is 4.

The phantom-step fix is preserved: start_time == -1 steps stay excluded, and the
surviving sequence is sorted by start_time, exactly as src/data.py does.
"""

from __future__ import annotations

import collections
import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import DATA_DIR, _norm_recipe, load_protocols

EXECUTION_TAGS = {"Preparation Error", "Measurement Error", "Timing Error",
                  "Technique Error", "Temperature Error"}
CHOICE_TAGS = {"Order Error", "Missing Step"}
OTHER_TAGS = {"Other"}


@dataclass
class Point:
    """One scored decision point, with the label of the step it precedes."""
    recording_id: str
    recipe: str
    person_id: int
    index: int                      # position in the kept, time-sorted sequence
    target_step_id: int
    target_start_time: float
    tags: frozenset
    y_execution: int                # the ruling's positive class
    y_any_error: int                # any has_errors, for contrast only
    done: list = field(default_factory=list)


def build_points():
    protocols = load_protocols()
    comp = json.load(open(os.path.join(DATA_DIR, "annotation_json",
                                       "complete_step_annotations.json")))
    ea = {r["recording_id"]: r for r in
          json.load(open(os.path.join(DATA_DIR, "annotation_json",
                                      "error_annotations.json")))}

    points = []
    for rec in comp.values():
        recipe = _norm_recipe(rec["activity_name"])
        proto = protocols.get(recipe)
        if proto is None:
            continue
        err = ea[rec["recording_id"]]
        assert len(err["step_annotations"]) == len(rec["steps"]), \
            f"{rec['recording_id']}: annotation arrays differ in length"

        # zip positionally, then apply the same two order fixes as data.py
        paired = list(zip(rec["steps"], err["step_annotations"]))
        for a, b in paired:
            assert a["step_id"] == b["step_id"] and abs(a["start_time"] - b["start_time"]) < 1e-6, \
                f"{rec['recording_id']}: positional misalignment at step {a['step_id']}"
        paired = [(a, b) for a, b in paired
                  if a["step_id"] in proto.steps and a["start_time"] != -1]
        paired.sort(key=lambda ab: ab[0]["start_time"])
        if len(paired) < 2:
            continue

        seq = [a["step_id"] for a, _ in paired]
        remaining = list(seq)
        done = []
        for i, (a, b) in enumerate(paired):
            if len(set(remaining)) >= 2:
                tags = frozenset(e["tag"] for e in (b.get("errors") or []))
                points.append(Point(
                    recording_id=rec["recording_id"], recipe=recipe,
                    person_id=rec["person_id"], index=i,
                    target_step_id=a["step_id"], target_start_time=a["start_time"],
                    tags=tags,
                    y_execution=int(bool(tags & EXECUTION_TAGS)),
                    y_any_error=int(bool(a.get("has_errors"))),
                    done=list(done),
                ))
            done.append(a["step_id"])
            remaining.remove(a["step_id"])
    return points


def summarize(points, label=""):
    folds = sorted({p.person_id for p in points})
    tag_counts = collections.Counter(t for p in points for t in p.tags)
    per_fold = {f: {"n": 0, "pos_exec": 0, "pos_any": 0, "order": 0} for f in folds}
    for p in points:
        d = per_fold[p.person_id]
        d["n"] += 1
        d["pos_exec"] += p.y_execution
        d["pos_any"] += p.y_any_error
        d["order"] += int("Order Error" in p.tags)

    L = [f"--- {label}",
         f"    decision points                 {len(points)}",
         f"    positives, execution-only       {sum(p.y_execution for p in points)}"
         f"  ({100*sum(p.y_execution for p in points)/len(points):.1f}%)",
         f"    positives, any has_errors       {sum(p.y_any_error for p in points)}"
         f"  ({100*sum(p.y_any_error for p in points)/len(points):.1f}%)",
         f"    Order Error points (in denom.)  {sum(1 for p in points if 'Order Error' in p.tags)}",
         "",
         f"    {'fold':<6}{'n':>7}{'pos_exec':>10}{'rate':>8}{'pos_any':>9}{'order':>7}"]
    degenerate = []
    for f in folds:
        d = per_fold[f]
        rate = d["pos_exec"] / d["n"] if d["n"] else 0.0
        flag = ""
        if d["pos_exec"] == 0 or d["pos_exec"] == d["n"]:
            flag = "  <-- DEGENERATE (single-class fold)"
            degenerate.append(f)
        elif d["pos_exec"] < 25:
            flag = "  <-- thin"
            degenerate.append(f)
        L.append(f"    {f:<6}{d['n']:>7}{d['pos_exec']:>10}{rate:>8.3f}"
                 f"{d['pos_any']:>9}{d['order']:>7}{flag}")
    L.append("")
    L.append("    per-tag counts (a point may carry several tags):")
    for t, c in tag_counts.most_common():
        cls = "POS" if t in EXECUTION_TAGS else ("choice" if t in CHOICE_TAGS else "other")
        L.append(f"      {t:<20}{c:>6}   -> {cls}")
    return "\n".join(L), degenerate, per_fold


def main():
    from run import PILOT_RECIPES
    points = build_points()
    out = {}
    blocks = []
    for name, sel in (("FULL (24 recipes)", None),
                      ("PILOT (5 recipes)", set(PILOT_RECIPES))):
        pts = [p for p in points if sel is None or p.recipe in sel]
        text, degen, per_fold = summarize(pts, name)
        blocks.append(text)
        out[name.split()[0].lower()] = {
            "n_points": len(pts),
            "pos_execution": sum(p.y_execution for p in pts),
            "pos_any_error": sum(p.y_any_error for p in pts),
            "order_points": sum(1 for p in pts if "Order Error" in p.tags),
            "per_fold": {str(k): v for k, v in per_fold.items()},
            "degenerate_or_thin_folds": degen,
            "tag_counts": dict(collections.Counter(t for p in pts for t in p.tags)),
        }
    text = "\n\n".join(blocks)
    print(text)
    os.makedirs("results", exist_ok=True)
    json.dump({"ruling": {"positive_tags": sorted(EXECUTION_TAGS),
                          "negative_tags": sorted(CHOICE_TAGS | OTHER_TAGS),
                          "other_policy": "negative, all 8, no hand-labelling",
                          "missing_step_on_performed": "negative by tag rule"},
               "labelling": "positional zip by array index; phantom steps excluded; "
                            "sequence sorted by start_time",
               "slices": out},
              open("results/labels_stage1.json", "w"), indent=2)
    open("results/labels_stage1.txt", "w").write(text + "\n")
    print("\nwrote results/labels_stage1.json")


if __name__ == "__main__":
    main()
