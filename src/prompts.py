"""Prompt templates, logged verbatim for reproducibility.

One template serves all four arms. The ONLY thing that varies between arms is the
SKILL_BLOCK substitution:
   Arm A -> ""                      (baseline, no skill)
   Arm B -> flat advice
   Arm C -> cue-conditioned
   Arm D -> scrambled-cue control

Everything else -- protocol text, completed-step history, candidate list, output
format -- is byte-identical across arms, so any difference in score is
attributable to the skill representation and nothing else.
"""

SYSTEM = (
    "You are predicting how a person actually performs a cooking recipe in a real kitchen. "
    "Recipes leave much of the ordering unspecified; experienced cooks follow conventions "
    "that the written recipe does not state. Rank the remaining steps by how likely each is "
    "to be the very next step this person performs."
)

USER_TEMPLATE = """RECIPE: {recipe}

{protocol}

STEPS THIS PERSON HAS ALREADY COMPLETED, in order:
{history}
{skill_block}
REMAINING STEPS (candidates for the next action):
{candidates}

Rank ALL {n_candidates} remaining step IDs from most to least likely to be the NEXT step this person performs.

Output ONLY a comma-separated list of the step IDs in rank order, most likely first. No other text.
Example format: 7, 3, 12, 1"""


def build_user_prompt(decision, protocol, skill_block: str) -> str:
    history = "\n".join(
        f"  {i+1}. [{sid}] {protocol.steps.get(sid, '')}" for i, sid in enumerate(decision.done)
    ) or "  (none yet - this is the first step)"
    candidates = "\n".join(
        f"  [{sid}] {protocol.steps.get(sid, '')}" for sid in decision.candidates
    )
    block = f"\n{skill_block}\n" if skill_block else "\n"
    return USER_TEMPLATE.format(
        recipe=decision.recipe,
        protocol=protocol.recipe_text(),
        history=history,
        skill_block=block,
        candidates=candidates,
        n_candidates=len(decision.candidates),
    )
