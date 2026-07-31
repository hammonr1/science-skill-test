"""Two pre-registration checks, both API-free.

A. CUE-LEAKAGE GUARD. The worry: if a cue's surface text overlaps the action's
   surface text, Arm C wins for a trivial reason -- a string matcher could pick
   the answer straight out of the cue, with no conditioning happening. We test
   exactly that: rank candidates by lexical similarity to the cue alone and see
   whether the true next action floats to the top.

B. SKILL-BUDGET CURVE. headroom.py used unlimited order statistics. A real skill
   list is a handful of rules. This measures how much of the headroom survives
   when the residual is compressed to the top-K rules per recipe -- i.e. how much
   an LLM arm could plausibly recover, which sets the effect size the API run
   needs to detect.
"""

from __future__ import annotations

import collections
import json
import random
import re
import statistics

from data import load_all
from headroom import SEED, reciprocal_rank


def tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z]+", s.lower())) - {
        "the", "a", "an", "of", "to", "and", "into", "in", "on", "with", "for", "it", "is",
    }


# ---------------------------------------------------------------- A. leakage

def cue_action_leakage(protocols, executions):
    """Build the cue/action pairs the authoring stage will use, then attack them.

    Cue at authoring granularity = the description of the just-completed step.
    Action = the description of the step actually chosen next.
    """
    rows = []
    for ex in executions:
        proto = protocols[ex.recipe]
        for prev, nxt in zip(ex.step_ids, ex.step_ids[1:]):
            rows.append((ex.recipe, proto.steps.get(prev, ""), proto.steps.get(nxt, ""), prev, nxt))

    jac = []
    for _, cue, act, _, _ in rows:
        tc, ta = tokens(cue), tokens(act)
        jac.append(len(tc & ta) / max(len(tc | ta), 1))

    # The actual attack: can lexical similarity to the cue alone pick the answer?
    _, _, decisions = load_all()
    rng = random.Random(SEED)
    rr_attack, top1_attack = [], []
    for d in decisions:
        if not d.done:
            continue
        proto = protocols[d.recipe]
        cue_t = tokens(proto.steps.get(d.done[-1], ""))
        cands = list(d.candidates)
        rng.shuffle(cands)
        ranked = sorted(
            cands,
            key=lambda s: -len(cue_t & tokens(proto.steps.get(s, ""))) / max(len(cue_t | tokens(proto.steps.get(s, ""))), 1),
        )
        rr_attack.append(reciprocal_rank(ranked, d.true_next))
        top1_attack.append(1.0 if ranked[0] == d.true_next else 0.0)

    rng2 = random.Random(SEED)
    rr_rand = []
    for d in decisions:
        if not d.done:
            continue
        c = list(d.candidates)
        rng2.shuffle(c)
        rr_rand.append(reciprocal_rank(c, d.true_next))

    return {
        "n_cue_action_pairs": len(rows),
        "mean_jaccard_cue_vs_action": statistics.mean(jac),
        "pairs_with_jaccard_over_0.5": sum(j > 0.5 for j in jac) / len(jac),
        "lexical_attack_mrr": statistics.mean(rr_attack),
        "random_mrr_same_subset": statistics.mean(rr_rand),
        "lexical_attack_top1": statistics.mean(top1_attack),
    }


# ------------------------------------------------------------ B. skill budget

def skill_budget_curve(executions, decisions, budgets=(3, 5, 8, 12, 20, 1000)):
    """Top-K residual rules per recipe, evaluated leave-one-participant-out."""
    persons = sorted({e.person_id for e in executions})
    by_person = collections.defaultdict(list)
    for d in decisions:
        by_person[d.person_id].append(d)

    out = {}
    for K in budgets:
        fold = []
        for held in persons:
            train = [e for e in executions if e.person_id != held]
            counts = collections.defaultdict(collections.Counter)
            for ex in train:
                for prev, nxt in zip(ex.step_ids, ex.step_ids[1:]):
                    counts[ex.recipe][(prev, nxt)] += 1
            # keep only the K most frequent (cue -> action) rules per recipe
            rules = {}
            for recipe, c in counts.items():
                keep = {}
                for (prev, nxt), n in c.most_common():
                    if prev in keep:
                        continue
                    if len(keep) >= K:
                        break
                    keep[prev] = (nxt, n)
                rules[recipe] = keep

            rng = random.Random(SEED + held)
            rrs = []
            for d in by_person[held]:
                cand = list(d.candidates)
                rng.shuffle(cand)
                fired = rules.get(d.recipe, {}).get(d.done[-1] if d.done else None)
                ready = set(d.ready)
                ranked = sorted(
                    cand,
                    key=lambda s: (
                        0 if (fired and s == fired[0]) else 1,
                        0 if s in ready else 1,
                    ),
                )
                rrs.append(reciprocal_rank(ranked, d.true_next))
            fold.append(statistics.mean(rrs))
        out[K] = {"mean": statistics.mean(fold), "folds": fold}
    return out


def run():
    protocols, executions, decisions = load_all()

    print("=" * 74)
    print("A. CUE-LEAKAGE GUARD")
    print("=" * 74)
    leak = cue_action_leakage(protocols, executions)
    for k, v in leak.items():
        print(f"  {k:<34} {v:.4f}" if isinstance(v, float) else f"  {k:<34} {v}")
    lift = leak["lexical_attack_mrr"] - leak["random_mrr_same_subset"]
    print(f"\n  lexical-attack lift over random: {lift:+.4f}")
    print("  VERDICT:", "PASS - cue text does not leak the action"
          if lift < 0.03 else "FAIL - coarsen the cue")

    print()
    print("=" * 74)
    print("B. SKILL-BUDGET CURVE (how much headroom survives compression)")
    print("=" * 74)
    curve = skill_budget_curve(executions, decisions)
    print(f"  {'K rules/recipe':<18}{'MRR':>10}{'  lift over protocol-only':>26}")
    base = json.load(open("results/headroom.json"))["mrr"]["2_protocol_only"]
    basem = statistics.mean(base)
    for K, v in curve.items():
        label = "unlimited" if K == 1000 else str(K)
        print(f"  {label:<18}{v['mean']:>10.4f}{v['mean'] - basem:>+22.4f}")

    json.dump({"leakage": leak, "skill_budget": curve}, open("results/diagnostics.json", "w"), indent=2)
    print("\nwrote results/diagnostics.json")


if __name__ == "__main__":
    run()
