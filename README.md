# Cue-conditioned skills vs. flat advice

Tests whether writing a learned skill as a cue→action conditional ("when you have
just finished X, do Y") makes an unseen operator's next action more predictable
than the same action content with the cue field deleted.

The question resists a direct answer. A kitchen session cannot be re-run under a
different skill, so the counterfactual — what this operator would have done given
advice Z — is never observed. Each recording is one draw from one operator, so the
skill's contribution cannot be recovered by comparing an execution to itself. The
test is displaced onto a frozen model: hold one participant out, condition on
skills mined from the other seven, and measure whether the held-out operator's
true next step gets easier to rank.

Data: [CaptainCook4D](https://github.com/CaptainCook4D/annotations) at `a8a920a` —
24 recipes, 8 participants, 384 recordings, 5,029 decision points, text only.

## Result

Pilot: 5 of 24 recipes, `claude-haiku-4-5`, 1,063 decisions per arm. MRR of the
true next step, leave-one-participant-out.

| arm | skill block | MRR | 95% CI | | contrast | Δ MRR | 95% CI | p (Holm) | folds + |
|---|---|---|---|---|---|---|---|---|---|
| A | none | 0.7115 | [0.6892, 0.7577] | | C − D | +0.0777 | [+0.0631, +0.0918] | 0.0000 | 8/8 |
| B | action text only | 0.7092 | [0.6865, 0.7569] | | C − B | +0.0166 | [+0.0018, +0.0302] | 0.0428 | 6/8 |
| C | cue → action | 0.7281 | [0.7067, 0.7669] | | B − A | −0.0019 | [−0.0036, +0.0000] | 0.0428 | 2/8 |
| D | scrambled cue | 0.6474 | [0.6215, 0.6983] | | | | | | |

Paired cluster bootstrap over participants, 10,000 resamples.

C − D is the effect to quote: positive in 8 of 8 folds, and the only contrast
whose two arms contain identical text (`results/final.json`). Pairing a cue with
the right action beats pairing it with the wrong one.

C − B is weak: just inside α=0.05 after correction, 6 of 8 folds, and at +0.0166
above the +0.0069 an oracle following the same K=8 rules achieves
(`results/power.json`) — so it is not measuring rule-following alone.

C − A is +0.0165, a difference of per-arm means rather than a pre-registered
contrast, so it carries no CI or correction. A lexical attack on cue text alone
buys +0.0138 over the task graph every arm sees (`results/diagnostics.json`) —
the same order of magnitude, so C − A is not evidence of cue conditioning. C − D
is immune, since C and D contain identical text.

B − A is a rounding error whose interval touches zero. An earlier run put it at
−0.0102 and read it as flat advice actively hurting; that vanished when two
step-order bugs were fixed, and the reading is withdrawn.

## Design

Arm D scrambles the cue field within the skill list — cue from skill *j* on the
action from skill *i* — so C and D carry the same cues, the same actions, and the
same line count, differing only in which cue goes with which action
(`src/skills.py`). Arm B deletes the cue, removing the mapping rather than
rearranging it, so C − B confounds representation with information content. C − D
holds content fixed and varies only the pairing.

## Reproduce

```bash
pip install anthropic
python src/run.py --stage precheck          # no API: headroom, leakage, power
python src/run.py --stage pilot --dry-run   # cost estimate
export ANTHROPIC_API_KEY=...
python src/run.py --stage pilot             # 4,252 calls, ~$5.18 est.
python src/stop_conditions.py pilot
python src/final_report.py pilot            # writes results/final.json
```

Temperature 0, fixed seeds, prompt templates logged into the results JSON.
Responses cache to `results/cache.sqlite`; re-runs cost nothing.

## Layout

- `src/data.py` — loads annotations; projects task graphs from recipe-local to global step ids.
- `src/skills.py` — mines (cue, action) pairs; renders arms B, C, D.
- `src/prompts.py` — the prompt template; only the skill block varies.
- `src/evaluate.py` — job construction, model calls, response cache, rank parsing.
- `src/stats.py` — paired cluster bootstrap, Holm correction.
- `src/headroom.py` — non-LLM ceiling on the transferable residual.
- `src/diagnostics.py` — cue-leakage guard, skill-budget curve.
- `src/power.py` — oracle effect against minimum detectable effect at n=8.
- `src/gate2_verify.py` — asserts the protocol/execution join is non-vacuous.
- `src/error_slice.py` — re-runs the headroom analysis with induced errors excluded.
- `src/stop_conditions.py` — pre-registered trip checks.
- `src/final_report.py` — emits `results/final.json`.
- `src/run.py` — entry point.

## Limitations

Cooking is not a wet lab. The step vocabulary is closed and the task graph is
supplied; a protocol whose steps cannot be enumerated in advance is a different
problem.

Skills are admitted by frequency, not lift. `mine_residual` keeps the K most
common (cue → action) transitions per recipe, so a frequent but uninformative rule
displaces a rare but diagnostic one.

Leakage is unresolved. Cue and action text share vocabulary (mean Jaccard 0.103),
and the +0.0138 conditional lift is twice the oracle K=8 skill effect of +0.0069
(`results/power.json`). Any contrast that does not hold cue text fixed inherits
it, which is why C − D is the quotable one.

The design is barely powered. The +0.0069 oracle ceiling sits against a minimum
detectable effect of 0.0061 at n=8 folds. A null here would not separate a false
hypothesis from an underpowered test.

Part of the residual is experimenter-induced. Participants were assigned to follow
recipes or to induce errors from designed categories, and Order Error is the
largest error class while the metric is step ordering. Excluding error recordings
drops the transferable headroom from +0.0510 to +0.0353 and from 8/8 to 7/8 folds
(`results/error_slice_full.json`) — the effect survives, inflated ~1.6×.

Two step-order bugs were found after the first pilot and fixed: 287 steps
annotated as skipped still held a sequence position, and 72 recordings were not
chronological. Both fabricated transitions, which is all the residual model reads.
Every figure above is post-fix; `git log` has the before and after.

The full 24-recipe run has not been executed. All figures are the 5-recipe pilot.

## License

MIT (`LICENSE`), covering the code in `src/`. The CaptainCook4D annotations
vendored under `data/` are redistributed under their upstream terms.
