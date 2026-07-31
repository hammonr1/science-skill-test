# Does a cue → action skill transfer better than flat advice?

A four-arm test of whether representing a learned skill as a CUE → ACTION
conditional makes an unseen operator's next action more predictable than
representing the same content as flat advice.

**Dataset:** [CaptainCook4D](https://github.com/CaptainCook4D/annotations)
annotations, pinned at commit `a8a920a`. Text-level only — no video is needed or
downloaded.

## The task

CaptainCook4D's task graphs are *partial* orders: at a typical point in a recipe
~8.7 of the ~9.1 remaining steps are legally performable. The written recipe
therefore barely constrains ordering, and which step an operator actually does
next is mostly convention the recipe never states. That gap is the residual a
skill is supposed to capture.

At each decision point the frozen model sees the recipe (steps + ordering
constraints), the steps completed so far, and all remaining steps, and ranks them.
Metric: **MRR** of the true next step, with top-1 logged alongside.

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
training participants; the 8th is only ever evaluated. `build_jobs` asserts no
participant overlap in either direction.

## Running

```bash
pip install anthropic
python src/run.py --stage precheck              # no API: power + leakage checks
python src/run.py --stage pilot --dry-run       # cost estimate
python src/run.py --stage pilot                 # 5 recipes,  ~4.5k calls
python src/run.py --stage full                  # 24 recipes, ~21k calls
```

Responses are cached in `results/cache.sqlite`, so runs are resumable and a
re-run costs nothing. Temperature 0, seeds fixed, prompt templates logged
verbatim into the results JSON.

## Pre-registered checks (all pass, see `results/`)

- **Power / headroom.** Residual statistics learned from training participants
  lift held-out-participant MRR from 0.386 (recipe alone) to 0.712 — +0.326,
  positive in 8/8 folds. There is a large operator-transferable residual to find.
- **Cue-leakage guard.** Mean Jaccard overlap between cue text and action text is
  0.003; no pair exceeds 0.5; a pure lexical-similarity attack on the cue scores
  −0.001 vs random. A cue cannot reveal its action by surface form.
- **Skill-budget curve.** At the K=8 rules/recipe the arms actually use, the
  compressed residual is worth +0.147 MRR — the effect size the LLM arms are
  powered to detect.

## Known deviation from the original design

No human-written tips exist anywhere in CaptainCook4D, so the requested human
ceiling arm is not constructible. `results/headroom.json` supplies a
**statistical ceiling** instead (unlimited residual model, 0.712 MRR).
