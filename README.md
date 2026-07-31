# Does a cue → action skill transfer better than flat advice?

A four-arm test of whether representing a learned skill as a CUE → ACTION
conditional makes an unseen operator's next action more predictable than
representing the same content as flat advice.

**Dataset:** [CaptainCook4D](https://github.com/CaptainCook4D/annotations)
annotations, pinned at commit `a8a920a`. Text-level only — no video is needed or
downloaded. Confirmed contents: **24 recipes, 8 participants, 384 recordings**
(220 containing errors), **5,313 decision points**.

Annotation schema (`complete_step_annotations.json`), verified directly:
`recording_id`, `activity_id`, `activity_name`, `person_id`, `environment`, and
`steps[]` of `{step_id, start_time, end_time, description, has_errors}`.

## Status: pre-registered, NOT yet run

The API-free pre-checks are complete and reproducible. **The four LLM arms have
not been run** — this environment has no Anthropic credential. `results/` contains
only the no-API diagnostics; there is no result about the hypothesis yet.

## The task

At each decision point the frozen model sees the recipe (steps + ordering
constraints), the steps completed so far, and all remaining steps, and ranks them.
Metric: **MRR** of the true next step, with top-1 logged alongside. MRR is primary
because candidate sets average ~8.9 and top-1 discards most of each ranking's
signal.

## Arms

| Arm | Skill block |
|-----|-------------|
| A | none (baseline) |
| B | flat advice — action text only |
| C | cue-conditioned — `When you have just finished <cue> → next do <action>` |
| D | scrambled cue — cue from skill *j* on action from skill *i* (within-field derangement) |

Arms B/C/D carry **identical action content**; only the representation differs.
The residual is mined deterministically into (cue, action) pairs so the arms
cannot differ in content — if a model free-wrote prose per arm, any C-vs-B gap
would be uninterpretable.

## Hold-out

Leave-one-participant-out over all 8 participants. Skills are authored from the 7
training participants; the 8th is only ever evaluated. `build_jobs` and
`power.paired_fold_deltas` assert no participant overlap in either direction.

## The ID-space bug (found before any API spend)

CaptainCook4D uses **two incompatible step-id spaces**: annotations are written in
a GLOBAL `step_idx` (1–350, unique across all recipes), while `task_graphs/*.json`
use RECIPE-LOCAL ids (`0=START, 1..N, N+1=END`). Broccoli stir fry's local `[1]` is
global `250`.

The first draft joined them by raw id. The join silently failed: every step
description in the prompt rendered **blank**, and the task-graph DAG was vacuous
(`predecessors` of an unmatched id is the empty set, so almost every step looked
"ready"). `src/data.py` now normalises everything into the global space, projecting
each task graph by exact step-text match — step text is unique within every
recipe's graph, so the map is well defined. Three recipes (dressedupmeatballs,
pinwheels, sautedmushrooms) genuinely repeat a step, so the projection is
many-to-one there; self-loops are dropped and candidate sets are deduplicated.

Every headline number changed:

| quantity | broken join | corrected |
|---|---|---|
| mean task-graph-ready steps | 8.67 of 9.12 | **2.48 of 8.92** |
| protocol-only MRR (held-out operator) | 0.386 | **0.713** |
| transferable residual headroom | +0.326 | **+0.058** |
| skill-budget lift at K=8 | +0.147 | **+0.017** |
| cue-leakage guard | PASS (vacuously — text was empty) | **marginal FAIL / conditional PASS** |

The recipe constrains ordering *far* more than the first draft implied, and the
residual a skill could carry is roughly 6× smaller.

## Pre-registered checks (`python src/run.py --stage precheck`, no API)

- **Headroom.** Residual statistics learned from training participants lift
  held-out-participant MRR from 0.713 (recipe alone) to 0.771 — +0.058, positive
  in 8/8 folds. A transferable residual exists, but it is small.
- **Cue-leakage guard.** Mean Jaccard overlap between cue and action text is 0.100.
  A lexical-similarity attack on the cue alone beats random by **+0.075** (marginal
  FAIL), but adds only **+0.018** on top of the task graph that every arm already
  sees (conditional PASS). The conditional number is the decision-relevant one —
  *but it is the same size as the entire oracle skill effect below*, which is a
  standing threat to interpreting any C-vs-B gap.
- **Power / go-no-go.** An **oracle** that applies the K=8 rules perfectly gains
  only **+0.0173 MRR**, against a minimum detectable effect of **0.0054** at n=8
  folds. Powered for a perfect rule-follower with ~3× margin; a real LLM realises
  some fraction of that, so the design is **marginal, not comfortable**.

## Running

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...                    # required for pilot/full
python src/run.py --stage precheck               # no API: headroom, leakage, power
python src/run.py --stage pilot --dry-run        # cost estimate
python src/run.py --stage pilot                  # 5 recipes,  4,468 calls, ~$5
python src/run.py --stage full                   # 24 recipes, 21,252 calls, ~$24
```

Costs are for `claude-haiku-4-5`. Responses are cached in `results/cache.sqlite`,
so runs are resumable and a re-run costs nothing. Temperature 0, seeds fixed,
prompt templates logged verbatim into the results JSON.

## Known deviations from the original design

- **No human ceiling.** No human-written tips exist anywhere in CaptainCook4D, so
  the requested human-ceiling arm is not constructible. `results/headroom.json`
  supplies a **statistical ceiling** instead (unlimited residual model, 0.771 MRR).
- **C-vs-B is an information ablation, not a pure representation contrast.**
  Deleting the cue field removes the cue→action *mapping*, not just its form, so
  Arm C strictly dominates Arm B in information. C>B is close to guaranteed; the
  load-bearing contrast is **C vs D**, where both arms carry the same cues and the
  same actions and differ only in whether the pairing is correct.
- **Arm B can contain duplicate lines** when two cues share an action, which
  faintly leaks frequency information. This is the faithful "cue field removed"
  ablation, so it is kept and noted.
