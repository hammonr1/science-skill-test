"""Power analysis / go-no-go, with zero API cost.

Before spending anything on LLM arms, establish that the quantity the experiment
is trying to detect actually exists. We decompose next-action predictability on
HELD-OUT participants into three levels:

  1. RANDOM        -- ranking candidates uniformly at random. The floor.
  2. PROTOCOL      -- ranking by the task graph alone (ready steps first).
                      This is what the written recipe explains: the Arm A analogue.
  3. PROTOCOL+RESIDUAL -- protocol, plus order statistics estimated from
                      TRAINING participants only. The extra lift here is the
                      transferable residual: an upper bound on what any skill,
                      however represented, could possibly contribute.

If level 3 does not beat level 2 on held-out operators, there is no operator-
transferable residual to represent, and the cue-vs-flat question is untestable on
this dataset -- no LLM arm could rescue it. That is a go/no-go, not a result
about the hypothesis itself.

Evaluated leave-one-participant-out, so every number is an unseen-operator number.
"""

from __future__ import annotations

import collections
import json
import os
import random
import statistics

from data import load_all

SEED = 20260731


def reciprocal_rank(ranking: list[int], truth: int) -> float:
    return 1.0 / (ranking.index(truth) + 1)


def mrr_random(n: int) -> float:
    """Expected reciprocal rank under a uniformly random ranking of n candidates."""
    return sum(1.0 / k for k in range(1, n + 1)) / n


class ResidualModel:
    """Order statistics learned from training participants only.

    Two backoff levels, both estimated per recipe:
      bigram   P(next = s | last completed step)
      unigram  mean normalised position of s across executions
    """

    def __init__(self, executions):
        self.bigram = collections.defaultdict(collections.Counter)
        self.position = collections.defaultdict(list)
        for ex in executions:
            n = len(ex.step_ids)
            prev = None
            for i, sid in enumerate(ex.step_ids):
                self.bigram[(ex.recipe, prev)][sid] += 1
                self.position[(ex.recipe, sid)].append(i / max(n - 1, 1))
                prev = sid
        self.mean_pos = {k: statistics.mean(v) for k, v in self.position.items()}

    def score(self, recipe: str, last: int | None, cand: int) -> tuple[float, float]:
        bg = self.bigram.get((recipe, last), {})
        total = sum(bg.values())
        p = bg.get(cand, 0) / total if total else 0.0
        # negative so that earlier-typical steps sort first
        pos = -self.mean_pos.get((recipe, cand), 0.5)
        return (p, pos)


def rank_random(d, rng, *_):
    c = list(d.candidates)
    rng.shuffle(c)
    return c


def rank_protocol(d, rng, *_):
    """Task-graph-ready steps first; ties broken at random (no residual info)."""
    ready = set(d.ready)
    c = list(d.candidates)
    rng.shuffle(c)
    return sorted(c, key=lambda s: 0 if s in ready else 1)


def rank_protocol_residual(d, rng, model):
    ready = set(d.ready)
    last = d.done[-1] if d.done else None
    c = list(d.candidates)
    rng.shuffle(c)
    return sorted(c, key=lambda s: (0 if s in ready else 1, -model.score(d.recipe, last, s)[0],
                                    -model.score(d.recipe, last, s)[1]))


def run():
    protocols, executions, decisions = load_all()
    persons = sorted({e.person_id for e in executions})
    by_person = collections.defaultdict(list)
    for d in decisions:
        by_person[d.person_id].append(d)

    rankers = {
        "1_random": rank_random,
        "2_protocol_only": rank_protocol,
        "3_protocol_plus_residual": rank_protocol_residual,
    }
    fold_scores = {k: [] for k in rankers}
    fold_top1 = {k: [] for k in rankers}

    for held in persons:
        train_ex = [e for e in executions if e.person_id != held]
        model = ResidualModel(train_ex)
        evald = by_person[held]
        assert all(d.person_id == held for d in evald)
        for name, fn in rankers.items():
            rng = random.Random(SEED + held)
            rrs = [reciprocal_rank(fn(d, rng, model), d.true_next) for d in evald]
            t1 = [1.0 if fn(random.Random(SEED + held + 991), rng, model) else 0 for d in evald[:0]]
            rng2 = random.Random(SEED + held)
            tops = [1.0 if fn(d, rng2, model)[0] == d.true_next else 0.0 for d in evald]
            fold_scores[name].append(statistics.mean(rrs))
            fold_top1[name].append(statistics.mean(tops))

    print("=" * 74)
    print("HEADROOM / POWER ANALYSIS  (leave-one-participant-out, 8 folds)")
    print("=" * 74)
    print(f"decision points evaluated : {len(decisions)}")
    print(f"mean candidate-set size   : {statistics.mean(len(d.candidates) for d in decisions):.2f}")
    print(f"analytic random MRR       : {statistics.mean(mrr_random(len(d.candidates)) for d in decisions):.4f}")
    print()
    print(f"{'level':<28}{'MRR':>10}{'  [min,max] over folds':>26}{'top-1':>10}")
    for name in rankers:
        s = fold_scores[name]
        t = fold_top1[name]
        print(f"{name:<28}{statistics.mean(s):>10.4f}   [{min(s):.4f},{max(s):.4f}]{'':>4}{statistics.mean(t):>10.4f}")

    prot = fold_scores["2_protocol_only"]
    resid = fold_scores["3_protocol_plus_residual"]
    deltas = [r - p for r, p in zip(resid, prot)]
    print()
    print("TRANSFERABLE RESIDUAL HEADROOM (level 3 - level 2), per held-out participant:")
    for p, d in zip(persons, deltas):
        print(f"  participant {p}: {d:+.4f}")
    print(f"  mean {statistics.mean(deltas):+.4f}   folds positive: {sum(x > 0 for x in deltas)}/{len(deltas)}")

    os.makedirs("results", exist_ok=True)
    json.dump(
        {
            "seed": SEED,
            "n_decisions": len(decisions),
            "folds": persons,
            "mrr": {k: v for k, v in fold_scores.items()},
            "top1": {k: v for k, v in fold_top1.items()},
            "residual_headroom_per_fold": deltas,
            "residual_headroom_mean": statistics.mean(deltas),
        },
        open("results/headroom.json", "w"),
        indent=2,
    )
    print("\nwrote results/headroom.json")


if __name__ == "__main__":
    run()
