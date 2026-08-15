"""Is the LLM experiment powered enough to be worth running?

After the id-space fix the transferable residual is much smaller than the first
draft believed, so this is now the go/no-go question. It has to be answered
BEFORE spending API budget, and it is answerable with no API at all.

Logic. The four-arm test's key contrast is C-vs-B (and C-vs-D). Whatever the
frozen model does, the information a K-rule skill list can add over the recipe
is bounded by an ORACLE that applies those same rules perfectly. So:

  effect_ceiling = MRR(protocol + top-K rules, applied perfectly)
                 - MRR(protocol alone)

measured leave-one-participant-out. The LLM arms cannot beat this ceiling; they
will realise some fraction of it. We then ask what effect the design could
actually detect: with 8 participant folds and the observed fold-to-fold spread,
what is the minimum detectable effect (MDE) of the paired cluster bootstrap?

If effect_ceiling < MDE, the experiment is underpowered BY CONSTRUCTION and no
amount of prompt engineering fixes it -- that is a design verdict, not a result
about the hypothesis.
"""

from __future__ import annotations

import collections
import json
import math
import os
import random
import statistics

from data import load_all
from headroom import SEED, reciprocal_rank


def topk_rules(train_ex, K):
    """Top-K (just-finished -> next) rules per recipe, from training operators."""
    counts = collections.defaultdict(collections.Counter)
    for ex in train_ex:
        for prev, nxt in zip(ex.step_ids, ex.step_ids[1:]):
            counts[ex.recipe][(prev, nxt)] += 1
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
    return rules


def rank_protocol(d, rng):
    ready = set(d.ready)
    c = list(d.candidates)
    rng.shuffle(c)
    return sorted(c, key=lambda s: 0 if s in ready else 1)


def rank_protocol_rules(d, rng, rules):
    """Oracle use of the skill list: a fired rule's action goes first."""
    ready = set(d.ready)
    fired = rules.get(d.recipe, {}).get(d.done[-1] if d.done else None)
    c = list(d.candidates)
    rng.shuffle(c)
    return sorted(c, key=lambda s: (0 if (fired and s == fired[0]) else 1,
                                    0 if s in ready else 1))


def paired_fold_deltas(decisions, executions, K):
    """Per-participant paired delta: oracle-with-rules minus protocol-alone."""
    persons = sorted({e.person_id for e in executions})
    by_person = collections.defaultdict(list)
    for d in decisions:
        by_person[d.person_id].append(d)

    deltas, base, with_rules = [], [], []
    for held in persons:
        train = [e for e in executions if e.person_id != held]
        assert all(e.person_id != held for e in train), "operator leak"
        rules = topk_rules(train, K)
        evald = by_person[held]
        assert all(d.person_id == held for d in evald), "operator leak"

        rng_a = random.Random(SEED + held)
        rng_b = random.Random(SEED + held)   # same stream -> paired tie-breaking
        pa = [reciprocal_rank(rank_protocol(d, rng_a), d.true_next) for d in evald]
        pb = [reciprocal_rank(rank_protocol_rules(d, rng_b, rules), d.true_next) for d in evald]
        base.append(statistics.mean(pa))
        with_rules.append(statistics.mean(pb))
        deltas.append(statistics.mean(y - x for x, y in zip(pa, pb)))
    return persons, base, with_rules, deltas


def mde(deltas, n_folds=8, alpha=0.05, power=0.8):
    """Minimum detectable effect for a paired test over participant folds."""
    sd = statistics.stdev(deltas)
    # two-sided alpha=.05, power=.80 -> (1.96 + 0.84) = 2.80
    return 2.80 * sd / math.sqrt(n_folds), sd


def run():
    protocols, executions, decisions = load_all()
    print("=" * 74)
    print("POWER / GO-NO-GO FOR THE LLM ARMS")
    print("=" * 74)

    out = {}
    print(f"{'K rules':<10}{'protocol':>10}{'+rules':>10}{'effect':>10}"
          f"{'fold SD':>10}{'MDE(n=8)':>11}{'powered?':>10}")
    for K in (3, 5, 8, 12, 20):
        persons, base, wr, deltas = paired_fold_deltas(decisions, executions, K)
        m, sd = mde(deltas)
        eff = statistics.mean(deltas)
        out[K] = {
            "protocol_mrr": statistics.mean(base),
            "with_rules_mrr": statistics.mean(wr),
            "effect_ceiling": eff,
            "fold_sd": sd,
            "mde_n8": m,
            "powered": eff > m,
            "per_fold": dict(zip(map(str, persons), deltas)),
        }
        print(f"{K:<10}{statistics.mean(base):>10.4f}{statistics.mean(wr):>10.4f}"
              f"{eff:>+10.4f}{sd:>10.4f}{m:>11.4f}{'YES' if eff > m else 'NO':>10}")

    print()
    print("READING: 'effect' is an ORACLE ceiling -- a perfect rule-follower.")
    print("The LLM arms realise some fraction of it, and the C-vs-B contrast is")
    print("smaller still because Arm B is not worth zero. If effect < MDE the")
    print("experiment cannot resolve the question on this dataset.")

    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/power.json", "w"), indent=2)
    print("\nwrote results/power.json")
    return out


if __name__ == "__main__":
    run()
