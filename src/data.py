"""CaptainCook4D data layer.

Loads the official annotation release (pinned commit a8a920a) and turns it into
the objects the experiment needs:

  Protocol   -- the written recipe: canonical step texts + task-graph DAG.
  Execution  -- one recording: an operator's actual ordering of those steps.
  Decision   -- one (context, candidates, true_next) prediction instance.

The central modelling claim of this file: the *protocol* underdetermines the
*execution*. The task graph is a partial order, so at most decision points
several steps are simultaneously legal. Which one the operator actually picks is
the residual -- the thing a skill could in principle predict.
"""

from __future__ import annotations

import collections
import glob
import json
import os
from dataclasses import dataclass, field

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

ANNOTATIONS_COMMIT = "a8a920a3293c4db27099a20ddbe3a3a9be1283e3"
ANNOTATIONS_REPO = "https://github.com/CaptainCook4D/annotations"


def _norm_recipe(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "")


@dataclass
class Protocol:
    """The written recipe: what the instructions actually specify."""

    recipe: str
    steps: dict[int, str]           # step_id -> canonical description
    edges: list[tuple[int, int]]    # (prereq, dependent)
    start_id: int
    end_id: int

    @property
    def predecessors(self) -> dict[int, set[int]]:
        preds: dict[int, set[int]] = collections.defaultdict(set)
        for a, b in self.edges:
            preds[b].add(a)
        return preds

    def ready(self, done: set[int], remaining: set[int]) -> list[int]:
        """Steps whose task-graph prerequisites are all satisfied."""
        preds = self.predecessors
        return sorted(s for s in remaining if preds[s] <= done)

    def recipe_text(self) -> str:
        """The protocol as a model would read it: steps plus ordering constraints."""
        lines = [f"  [{sid}] {txt}" for sid, txt in sorted(self.steps.items())
                 if sid not in (self.start_id, self.end_id)]
        preds = self.predecessors
        cons = []
        for sid in sorted(self.steps):
            if sid in (self.start_id, self.end_id):
                continue
            p = sorted(x for x in preds[sid] if x != self.start_id)
            if p:
                cons.append(f"  [{sid}] requires: {', '.join('[%d]' % x for x in p)}")
        out = "STEPS:\n" + "\n".join(lines)
        if cons:
            out += "\n\nORDERING CONSTRAINTS (from the recipe; anything not listed is free order):\n" + "\n".join(cons)
        return out


@dataclass
class Execution:
    """One recording: one operator performing one recipe."""

    recording_id: str
    recipe: str
    person_id: int
    environment: int
    step_ids: list[int]                     # in the order actually performed
    descriptions: dict[int, str]
    durations: dict[int, float]
    has_errors: dict[int, bool]

    @property
    def is_error_session(self) -> bool:
        return any(self.has_errors.values())


@dataclass
class Decision:
    """One next-action prediction instance."""

    recording_id: str
    recipe: str
    person_id: int
    index: int                  # position within the execution
    done: list[int]             # steps completed so far, in order
    candidates: list[int]       # steps not yet performed
    true_next: int
    ready: list[int]            # subset of candidates legal under the task graph
    meta: dict = field(default_factory=dict)


def load_protocols() -> dict[str, Protocol]:
    protocols = {}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "task_graphs", "*.json"))):
        key = os.path.basename(path)[:-5]
        g = json.load(open(path))
        steps = {int(k): v for k, v in g["steps"].items()}
        start = min(steps)
        end = max(steps)
        protocols[key] = Protocol(
            recipe=key,
            steps=steps,
            edges=[(int(a), int(b)) for a, b in g["edges"]],
            start_id=start,
            end_id=end,
        )
    return protocols


def load_executions() -> list[Execution]:
    raw = json.load(open(os.path.join(DATA_DIR, "annotation_json", "complete_step_annotations.json")))
    out = []
    for rec in raw.values():
        steps = rec["steps"]
        out.append(
            Execution(
                recording_id=rec["recording_id"],
                recipe=_norm_recipe(rec["activity_name"]),
                person_id=rec["person_id"],
                environment=rec["environment"],
                step_ids=[s["step_id"] for s in steps],
                descriptions={s["step_id"]: s["description"] for s in steps},
                durations={s["step_id"]: round(s["end_time"] - s["start_time"], 2) for s in steps},
                has_errors={s["step_id"]: bool(s.get("has_errors")) for s in steps},
            )
        )
    return sorted(out, key=lambda e: e.recording_id)


def build_decisions(ex: Execution, proto: Protocol, min_candidates: int = 2) -> list[Decision]:
    """Every point in an execution where >= min_candidates steps remain.

    Candidates are ALL remaining steps, not just task-graph-ready ones. Filtering
    to ready-only would bake the protocol into the harness and shrink the task;
    instead the protocol is given to every arm in the prompt, so all arms see the
    same constraints and the skill is the only thing that varies.
    """
    decisions = []
    done: list[int] = []
    remaining = list(ex.step_ids)
    for i, sid in enumerate(ex.step_ids):
        if len(remaining) >= min_candidates:
            decisions.append(
                Decision(
                    recording_id=ex.recording_id,
                    recipe=ex.recipe,
                    person_id=ex.person_id,
                    index=i,
                    done=list(done),
                    candidates=sorted(remaining),
                    true_next=sid,
                    ready=proto.ready({proto.start_id, *done}, set(remaining)),
                )
            )
        done.append(sid)
        remaining.remove(sid)
    return decisions


def load_all(min_candidates: int = 2):
    protocols = load_protocols()
    executions = [e for e in load_executions() if e.recipe in protocols]
    decisions = []
    for ex in executions:
        decisions.extend(build_decisions(ex, protocols[ex.recipe], min_candidates))
    return protocols, executions, decisions


if __name__ == "__main__":
    protocols, executions, decisions = load_all()
    persons = sorted({e.person_id for e in executions})
    print(f"annotations commit : {ANNOTATIONS_COMMIT}")
    print(f"recipes (protocols): {len(protocols)}")
    print(f"executions         : {len(executions)}")
    print(f"participants       : {len(persons)} -> {persons}")
    print(f"error sessions     : {sum(e.is_error_session for e in executions)}")
    print(f"decision points    : {len(decisions)}")
    import statistics
    print(f"mean candidates    : {statistics.mean(len(d.candidates) for d in decisions):.2f}")
    print(f"mean ready         : {statistics.mean(len(d.ready) for d in decisions):.2f}")
