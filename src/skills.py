"""Residual mining and skill authoring.

Design commitment that makes the four arms comparable: the residual is mined
DETERMINISTICALLY into (cue, action) pairs, and the model's job is only to render
those pairs as natural-language skills. Arms B/C/D therefore carry *identical
action content* and differ only in representation:

  C  cue-conditioned : "When <cue>, <action>."
  B  flat advice     : "<action>."                      (cue field deleted)
  D  scrambled cue   : "When <cue_j>, <action_i>."       (i != j, within-field)

If instead we let a model free-write prose per arm, the arms would differ in
content as well as form, and any C-vs-B gap would be uninterpretable.

Mining: for each recipe, over TRAINING participants only, count (just-completed
step -> next step) transitions. Keep the K most frequent, one rule per cue.
"cue" is an observable condition (which step the operator just finished);
"action" is the step they chose next among the several the recipe permits.
"""

from __future__ import annotations

import collections
import random
from dataclasses import dataclass


@dataclass
class Skill:
    recipe: str
    cue_step: int
    action_step: int
    cue_text: str
    action_text: str
    support: int
    total: int

    @property
    def confidence(self) -> float:
        return self.support / self.total if self.total else 0.0

    def render(self, arm: str, scrambled_cue_text: str | None = None) -> str:
        if arm == "B":
            return f"- {self.action_text}"
        if arm == "C":
            return f"- When you have just finished: {self.cue_text} -> next do: {self.action_text}"
        if arm == "D":
            return f"- When you have just finished: {scrambled_cue_text} -> next do: {self.action_text}"
        raise ValueError(arm)


def mine_residual(executions, protocols, recipe: str, k: int = 8) -> list[Skill]:
    """Top-k (cue -> action) rules for one recipe from the given executions."""
    trans = collections.Counter()
    cue_tot = collections.Counter()
    for ex in executions:
        if ex.recipe != recipe:
            continue
        for prev, nxt in zip(ex.step_ids, ex.step_ids[1:]):
            trans[(prev, nxt)] += 1
            cue_tot[prev] += 1

    proto = protocols[recipe]
    best: dict[int, tuple[int, int]] = {}
    for (prev, nxt), n in trans.most_common():
        if prev not in best:
            best[prev] = (nxt, n)

    ranked = sorted(best.items(), key=lambda kv: -kv[1][1])[:k]
    skills = []
    for cue_step, (action_step, n) in ranked:
        skills.append(
            Skill(
                recipe=recipe,
                cue_step=cue_step,
                action_step=action_step,
                cue_text=proto.steps.get(cue_step, f"step {cue_step}"),
                action_text=proto.steps.get(action_step, f"step {action_step}"),
                support=n,
                total=cue_tot[cue_step],
            )
        )
    return skills


def scramble_cues(skills: list[Skill], seed: int) -> list[str]:
    """Within-field derangement: cue from skill j attached to action from skill i.

    Not a whole-skill shuffle -- the point is to keep every cue and every action
    present in the prompt exactly once, and destroy only the PAIRING. If Arm C
    beats Arm D, the lift came from which cue goes with which action, not from
    the presence of cue-shaped text.
    """
    n = len(skills)
    if n < 2:
        return [s.cue_text for s in skills]
    rng = random.Random(seed)
    idx = list(range(n))
    for _ in range(1000):
        rng.shuffle(idx)
        if all(i != j for i, j in enumerate(idx)):
            break
    else:  # pragma: no cover - fallback derangement
        idx = [(i + 1) % n for i in range(n)]
    return [skills[j].cue_text for j in idx]


def render_skill_block(skills: list[Skill], arm: str, seed: int = 0) -> str:
    if arm == "A" or not skills:
        return ""
    if arm == "D":
        scr = scramble_cues(skills, seed)
        body = "\n".join(s.render("D", scr[i]) for i, s in enumerate(skills))
    else:
        body = "\n".join(s.render(arm) for s in skills)
    header = {
        "B": "PRACTITIONER TIPS (observed from experienced operators):",
        "C": "PRACTITIONER TIPS (observed from experienced operators):",
        "D": "PRACTITIONER TIPS (observed from experienced operators):",
    }[arm]
    return f"{header}\n{body}"
