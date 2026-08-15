"""Stage 2: binary execution-error anticipation, covered points only.

Task: given context STRICTLY PRIOR to a step instance, predict the probability
that the instance will be executed with an execution-quality error
({Preparation, Measurement, Timing, Technique, Temperature}).

LEAKAGE BOUNDARY (hard rule 4). The prompt is assembled so that nothing observed
at or after the target step's start_time can enter it:

  included   the recipe protocol (static, authored before any execution)
             the ordered list of steps COMPLETED BEFORE the target, which by
               construction all have start_time < target.start_time
             the identity and protocol text of the target step -- this is the
               question being asked, not an observation of the outcome
             the remaining steps, listed in ascending step-id order, NOT in the
               order the operator went on to perform them
             the arm's skill block, mined from TRAINING participants only

  excluded   the target step's has_errors flag, error tags, error descriptions,
             and modified_description
             the target step's end_time and duration
             every step after the target, and every timestamp at all
             any ordering information about the operator's future choices

Durations are never read. No timestamp appears anywhere in the prompt, so the
model cannot infer that a step ran long. Enforcement is asserted in code by
`assert_no_target_leakage`, which scans each rendered prompt for the target's
outcome strings.

Metric: Brier primary (decomposable per point, so pairing stays at the decision
point), AUROC secondary (fold-level). Mean and SD of the predicted probability
are reported per arm to show whether the model discriminates at all or sits at
the base rate.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import load_protocols
from evaluate import Cache
from labels import EXECUTION_TAGS, build_points
from library_churn import load_records
from library_variants import mine

ARMS = ["A", "B", "C", "D", "E"]
ARM_LABEL = {"A": "no skill (baseline)", "B": "flat advice",
             "C": "cue-conditioned", "D": "scrambled cue (control)",
             "E": "placebo: other recipe (control)"}
SEED = 7
MODEL = "claude-haiku-4-5-20251001"

SYSTEM = (
    "You are predicting how a person performs a cooking recipe in a real kitchen. "
    "You will be told which step they are about to perform next. Estimate the "
    "probability that they will perform that step with an execution-quality error: "
    "a preparation, measurement, timing, technique, or temperature mistake. "
    "This is about HOW WELL the step is carried out, not about which step they choose. "
    "If practitioner tips are shown, treat them as a general caution list compiled from "
    "other people's earlier sessions. They are not evidence about this particular person "
    "or this particular step, and their presence does not by itself make an error likely."
)

USER_TEMPLATE = """RECIPE: {recipe}

{protocol}

STEPS THIS PERSON HAS ALREADY COMPLETED, in order:
{history}
{skill_block}
THE STEP THEY ARE ABOUT TO PERFORM NEXT:
  [{target_id}] {target_text}

STEPS STILL REMAINING AFTER THIS ONE (listed in recipe order, not in the order they will be done):
{remaining}

What is the probability that this person performs the step above with an
execution-quality error (preparation, measurement, timing, technique, or temperature)?

Output ONLY an integer from 0 to 100. No other text.
Example format: 35"""


def scramble(rules, seed):
    """Within-field derangement of the cue field. Same control as the MRR arms."""
    import random
    n = len(rules)
    if n < 2:
        return [r["cue_step"] for r in rules]
    rng = random.Random(seed)
    idx = list(range(n))
    for _ in range(1000):
        rng.shuffle(idx)
        if all(i != j for i, j in enumerate(idx)):
            break
    else:
        idx = [(i + 1) % n for i in range(n)]
    return [rules[j]["cue_step"] for j in idx]


def render_block(rules, arm, protocol, seed=SEED):
    """Arm E is rendered by the caller with donor rules + the donor protocol,
    so it reaches here as an ordinary 'C' render: structurally identical to the
    treatment, topically irrelevant to the step being asked about."""
    if arm == "E":
        arm = "C"
    if arm == "A" or not rules:
        return ""
    scr = scramble(rules, seed) if arm == "D" else None
    lines = []
    for i, r in enumerate(rules):
        act = f"at [{r['target_step']}] {protocol.steps.get(r['target_step'],'')}, watch for: {r['intervention']}"
        if arm == "B":
            lines.append(f"- watch for: {r['intervention']}")
        elif arm == "C":
            cue = protocol.steps.get(r["cue_step"], f"step {r['cue_step']}")
            lines.append(f"- When you have just finished: {cue} -> {act}")
        else:
            cue = protocol.steps.get(scr[i], f"step {scr[i]}")
            lines.append(f"- When you have just finished: {cue} -> {act}")
    return "PRACTITIONER TIPS (observed from experienced operators):\n" + "\n".join(lines)


def build_prompt(point, protocol, block):
    history = "\n".join(f"  {i+1}. [{s}] {protocol.steps.get(s,'')}"
                        for i, s in enumerate(point.done)) or "  (none yet)"
    done = set(point.done)
    remaining = sorted(s for s in protocol.steps if s not in done and s != point.target_step_id)
    rem = "\n".join(f"  [{s}] {protocol.steps.get(s,'')}" for s in remaining) or "  (none)"
    return USER_TEMPLATE.format(
        recipe=point.recipe, protocol=protocol.recipe_text(), history=history,
        skill_block=f"\n{block}\n" if block else "\n",
        target_id=point.target_step_id,
        target_text=protocol.steps.get(point.target_step_id, ""),
        remaining=rem)


def assert_no_target_leakage(prompt, point, err_lookup):
    """No outcome string for the TARGET instance may appear in the prompt."""
    tokens = err_lookup.get((point.recording_id, point.index), [])
    low = prompt.lower()
    for t in tokens:
        t = (t or "").strip().lower()
        if len(t) > 12 and t in low:
            raise AssertionError(
                f"target outcome text leaked into prompt for "
                f"{point.recording_id}#{point.index}: {t[:60]!r}")
    for tag in EXECUTION_TAGS:
        if tag.lower() in low:
            raise AssertionError(f"error tag {tag!r} leaked into prompt")
    assert not re.search(r"\bstart_time\b|\bend_time\b|has_errors", low), "raw field leaked"


def parse_prob(text):
    m = re.search(r"\d+", text or "")
    if not m:
        return None
    return max(0.0, min(100.0, float(m.group()))) / 100.0


def call(client, cache, user, max_retries=5):
    k = hashlib.sha256(f"{MODEL}\x00{SYSTEM}\x00{user}".encode()).hexdigest()
    hit = cache.get(k)
    if hit is not None:
        return hit, True
    d = 2.0
    for a in range(max_retries):
        try:
            r = client.messages.create(model=MODEL, max_tokens=24, temperature=0,
                                       system=SYSTEM,
                                       messages=[{"role": "user", "content": user}])
            t = r.content[0].text if r.content else ""
            cache.put(k, t)
            return t, False
        except Exception:
            if a == max_retries - 1:
                raise
            time.sleep(d); d *= 2


def brier(ps, ys):
    return statistics.mean((p - y) ** 2 for p, y in zip(ps, ys))


def auroc(ps, ys):
    pos = [p for p, y in zip(ps, ys) if y == 1]
    neg = [p for p, y in zip(ps, ys) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def covered_points(fold, protocols, records, points):
    """Points in `fold` where a rule matches both cue and target (no floor)."""
    train = [r for r in records if r["person"] != fold]
    assert all(r["person"] != fold for r in train), "operator leak in authoring set"
    lib_by_recipe = {rc: mine(train, rc, "none") for rc in sorted(protocols)}
    by_target = collections.defaultdict(list)
    for rc, rs in lib_by_recipe.items():
        for r in rs:
            by_target[(rc, r["target_step"])].append(r)
    out = []
    for p in points:
        if p.person_id != fold:
            continue
        rs = by_target.get((p.recipe, p.target_step_id))
        cue = p.done[-1] if p.done else None
        if rs and any(r["cue_step"] == cue for r in rs):
            out.append(p)
    return out, lib_by_recipe


def main(fold=6, workers=12):
    import anthropic
    protocols = load_protocols()
    _, records = load_records()
    points = build_points()

    # target-instance outcome strings, for the leakage assertion
    err_lookup = {}
    ea = {r["recording_id"]: r for r in json.load(open(
        "data/annotation_json/error_annotations.json"))}
    idx = collections.defaultdict(int)
    for p in points:
        rec = ea[p.recording_id]
        for a in rec["step_annotations"]:
            if a["step_id"] == p.target_step_id and abs(a["start_time"] - p.target_start_time) < 1e-6:
                toks = [e.get("description") for e in (a.get("errors") or [])]
                toks.append(a.get("modified_description"))
                err_lookup[(p.recording_id, p.index)] = [t for t in toks if t]
                break

    pts, lib_by_recipe = covered_points(fold, protocols, records, points)
    print(f"fold {fold}: {len(pts)} covered points, {sum(p.y_execution for p in pts)} positives")

    # Arm E donor: deterministic rotation over recipes that actually have rules.
    # The donor library is rendered against the DONOR's protocol, so arm E is
    # arm C's template filled with real but topically irrelevant content.
    stocked = sorted(rc for rc, rs in lib_by_recipe.items() if rs)
    donor_of = {rc: stocked[(i + 1) % len(stocked)] for i, rc in enumerate(stocked)}

    block_tokens = collections.defaultdict(list)
    jobs = []
    for p in pts:
        proto = protocols[p.recipe]
        base = None
        for arm in ARMS:
            if arm == "E":
                dr = donor_of.get(p.recipe, p.recipe)
                block = render_block(lib_by_recipe[dr], "E", protocols[dr])
            else:
                block = render_block(lib_by_recipe[p.recipe], arm, proto)
            block_tokens[arm].append(len(block.split()))
            u = build_prompt(p, proto, block)
            assert_no_target_leakage(u, p, err_lookup)
            # arms must be byte-identical outside the skill block
            cut = u.index("PRACTITIONER TIPS") if "PRACTITIONER TIPS" in u else u.index("THE STEP THEY ARE ABOUT")
            residue = u[:cut].rstrip() + "\n" + u[u.index("THE STEP THEY ARE ABOUT"):]
            if base is None:
                base = residue
            assert residue == base, f"arms differ outside skill block at {p.recording_id}#{p.index}"
            jobs.append({"arm": arm, "point": p, "user": u})

    client = anthropic.Anthropic()
    cache = Cache()
    done = [0]; lock = threading.Lock()

    def work(j):
        t, cached = call(client, cache, j["user"])
        pr = parse_prob(t)
        with lock:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  {done[0]}/{len(jobs)}", flush=True)
        return {"arm": j["arm"], "recording_id": j["point"].recording_id,
                "index": j["point"].index, "y": j["point"].y_execution,
                "p": pr, "cached": cached, "raw": (t or "").strip()[:20]}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(work, jobs))

    bad = [r for r in rows if r["p"] is None]
    per_arm = {}
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm and r["p"] is not None]
        ps = [r["p"] for r in rs]; ys = [r["y"] for r in rs]
        per_arm[arm] = {
            "label": ARM_LABEL[arm], "n": len(rs),
            "brier": brier(ps, ys), "auroc": auroc(ps, ys),
            "p_mean": statistics.mean(ps), "p_sd": statistics.stdev(ps) if len(ps) > 1 else 0.0,
            "p_min": min(ps), "p_max": max(ps),
            "block_tokens_mean": statistics.mean(block_tokens[arm]),
            "n_unparseable": sum(1 for r in rows if r["arm"] == arm and r["p"] is None),
        }
    base_rate = statistics.mean(p.y_execution for p in pts)

    L = ["=" * 84,
         f"STAGE 2 DRY RUN -- fold {fold}, covered points only, UNDERPOWERED BY DESIGN",
         "=" * 84,
         f"  covered points {len(pts)}   positives {sum(p.y_execution for p in pts)}"
         f"   base rate {base_rate:.3f}",
         f"  model {MODEL}   temp 0   seed {SEED}   metric Brier (primary), AUROC (secondary)",
         "",
         f"{'arm':<5}{'description':<28}{'blk tok':>9}{'Brier':>9}{'AUROC':>8}"
         f"{'p_mean':>9}{'p_sd':>8}"]
    for arm in ARMS:
        a = per_arm[arm]
        L.append(f"{arm:<5}{a['label']:<28}{a['block_tokens_mean']:>9.1f}"
                 f"{a['brier']:>9.4f}{a['auroc']:>8.3f}"
                 f"{a['p_mean']:>9.3f}{a['p_sd']:>8.3f}")
    ce = per_arm["C"]["brier"] - per_arm["E"]["brier"]
    cd = per_arm["C"]["brier"] - per_arm["D"]["brier"]
    ca = per_arm["C"]["brier"] - per_arm["A"]["brier"]
    ba = per_arm["B"]["brier"] - per_arm["A"]["brier"]
    L += ["",
          f"  C - D  Brier {cd:+.4f}   (negative favours C: lower Brier is better)",
          f"  C - E  Brier {ce:+.4f}   (placebo: same template, other recipe)",
          f"  C - A  Brier {ca:+.4f}",
          f"  B - A  Brier {ba:+.4f}",
          f"  always-predict-base-rate Brier = {base_rate*(1-base_rate):.4f}",
          "",
          "  UNDERPOWERED: single fold, covered points only. No CI, no MDE, no",
          "  inferential claim. Directional read only."]
    if bad:
        L.append(f"  WARNING {len(bad)} unparseable responses")
    text = "\n".join(L)
    print("\n" + text)

    os.makedirs("results", exist_ok=True)
    json.dump({"stage": "stage2_dry_run", "fold": fold, "model": MODEL, "seed": SEED,
               "scoring": "covered points only (cue+target match), no support floor",
               "underpowered": True, "n_points": len(pts),
               "n_positives": sum(p.y_execution for p in pts), "base_rate": base_rate,
               "per_arm": per_arm, "brier_C_minus_D": cd, "brier_C_minus_E": ce, "brier_C_minus_A": ca,
               "brier_B_minus_A": ba,
               "baseline_brier_predict_base_rate": base_rate * (1 - base_rate),
               "n_calls": len(jobs), "n_cached": sum(1 for r in rows if r["cached"]),
               "system_prompt": SYSTEM, "user_template": USER_TEMPLATE},
              open(f"results/anticipate_fold{fold}.json", "w"), indent=2)
    json.dump(rows, open(f"results/anticipate_fold{fold}_rows.json", "w"), indent=2)
    open(f"results/anticipate_fold{fold}.txt", "w").write(text + "\n")
    print(f"\nwrote results/anticipate_fold{fold}.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
