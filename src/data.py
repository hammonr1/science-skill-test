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

ID SPACES (this bit is load-bearing and was wrong in the first draft).
CaptainCook4D uses two different step-id spaces:

  * ``annotation_json/*``      -- GLOBAL step_idx, 1..350, unique across all 24
                                  recipes. This is what recordings are written in.
  * ``task_graphs/<recipe>``   -- RECIPE-LOCAL ids, 0=START, 1..N, N+1=END.

They are NOT interchangeable: broccolistirfry's local [1] is global 250. Joining
them by raw id silently produces empty step descriptions and a vacuous DAG, so
everything here is normalised into the GLOBAL space, with the task graph
projected onto it by exact step-text match. Step text is unique within every
recipe's task graph, so the match is well defined.

Three recipes (dressedupmeatballs, pinwheels, sautedmushrooms) legitimately
REPEAT a step -- the same action twice in one recipe. The task graph gives each
occurrence its own local node; the global index gives them one id. The
projection is therefore many-to-one, and a repeated step can appear twice in one
recording's sequence.
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
    """The written recipe, in GLOBAL step-id space."""

    recipe: str
    steps: dict[int, str]           # global step_idx -> canonical description
    edges: list[tuple[int, int]]    # (prereq, dependent), global ids
    roots: set[int]                 # steps with no prerequisite

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
        lines = [f"  [{sid}] {txt}" for sid, txt in sorted(self.steps.items())]
        preds = self.predecessors
        cons = []
        for sid in sorted(self.steps):
            p = sorted(preds[sid])
            if p:
                cons.append(f"  [{sid}] requires: {', '.join('[%d]' % x for x in p)}")
        out = "STEPS:\n" + "\n".join(lines)
        if cons:
            out += ("\n\nORDERING CONSTRAINTS (from the recipe; anything not listed "
                    "is free order):\n" + "\n".join(cons))
        return out


@dataclass
class Execution:
    """One recording: one operator performing one recipe."""

    recording_id: str
    recipe: str
    person_id: int
    environment: int
    step_ids: list[int]                     # global ids, in the order performed
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
    candidates: list[int]       # distinct steps not yet performed
    true_next: int
    ready: list[int]            # subset of candidates legal under the task graph
    meta: dict = field(default_factory=dict)


def load_step_descriptions() -> dict[int, str]:
    raw = json.load(open(os.path.join(DATA_DIR, "annotation_json", "step_idx_description.json")))
    return {int(k): v for k, v in raw.items()}


def _activity_global_ids() -> dict[str, set[int]]:
    """recipe -> set of global step ids that recipe is made of."""
    act = json.load(open(os.path.join(DATA_DIR, "annotation_json", "activity_idx_step_idx.json")))
    comp = json.load(open(os.path.join(DATA_DIR, "annotation_json", "complete_step_annotations.json")))
    aid_to_recipe = {r["activity_id"]: _norm_recipe(r["activity_name"]) for r in comp.values()}
    out = {}
    for aid, seq in act.items():
        recipe = aid_to_recipe.get(int(aid))
        if recipe:
            out[recipe] = set(seq)
    return out


def load_protocols() -> dict[str, Protocol]:
    """Task graphs projected from recipe-local ids into the global step space."""
    desc = load_step_descriptions()
    recipe_ids = _activity_global_ids()
    protocols = {}

    for path in sorted(glob.glob(os.path.join(DATA_DIR, "task_graphs", "*.json"))):
        recipe = os.path.basename(path)[:-5]
        g = json.load(open(path))
        local = {int(k): v for k, v in g["steps"].items()}

        # global ids belonging to this recipe, keyed by their text
        gids = recipe_ids.get(recipe, set())
        text_to_global = {desc[i]: i for i in gids}

        # local node -> global id, by exact step text (many-to-one for repeats)
        l2g, dropped = {}, []
        for lid, txt in local.items():
            if txt in ("START", "END"):
                continue
            gid = text_to_global.get(txt)
            if gid is None:
                dropped.append((lid, txt))
            else:
                l2g[lid] = gid
        if dropped:
            raise ValueError(f"{recipe}: task-graph steps absent from annotations: {dropped}")

        # project edges, contracting START/END and dropping self-loops created
        # by the many-to-one repeat mapping
        edges = set()
        for a, b in g["edges"]:
            ga, gb = l2g.get(int(a)), l2g.get(int(b))
            if ga is None or gb is None or ga == gb:
                continue
            edges.add((ga, gb))

        steps = {i: desc[i] for i in sorted(l2g.values())}
        preds = {b for _, b in edges}
        protocols[recipe] = Protocol(
            recipe=recipe,
            steps=steps,
            edges=sorted(edges),
            roots={s for s in steps if s not in preds},
        )
    return protocols


class UnprojectedStepError(RuntimeError):
    """Raised when a recording contains a step absent from its projected protocol."""


def load_executions(protocols: dict[str, Protocol],
                    allow_dropped: int = 0) -> list[Execution]:
    """Load recordings, dropping any whose steps do not fully project.

    POLICY (confirmed, Gate 2): if a recording contains a step that does not
    project into the protocol's global id space, the WHOLE RECORDING is dropped
    -- never the individual step. Dropping an interior step would silently splice
    its neighbours together and fabricate a (cue -> action) transition that never
    happened, corrupting exactly the quantity this experiment mines.

    The policy is asserted rather than merely available: ``allow_dropped``
    defaults to 0, so if the data ever stops projecting cleanly this raises
    instead of quietly shrinking the sample. It must stay visible.
    """
    raw = json.load(open(os.path.join(DATA_DIR, "annotation_json", "complete_step_annotations.json")))
    out, dropped = [], []
    for rec in raw.values():
        recipe = _norm_recipe(rec["activity_name"])
        proto = protocols.get(recipe)
        if proto is None:
            continue
        bad = [s["step_id"] for s in rec["steps"] if s["step_id"] not in proto.steps]
        if bad:
            dropped.append({"recording_id": rec["recording_id"], "recipe": recipe,
                            "unprojected_step_ids": sorted(set(bad))})
            continue
        # Two order fixes. The residual model and the next-step metric read
        # nothing but step order, so both of these corrupt the quantity measured.
        #   1. 287 steps carry start_time == -1: annotated as skipped, never
        #      performed, yet occupying a position in the step array (141
        #      recordings, 85 with the phantom interior). A transition into or
        #      out of a step that never happened is fabricated.
        #   2. In 72 recordings the array order is not chronological (83 adjacent
        #      inversions); array order was being taken as execution order.
        steps = [s for s in rec["steps"] if s["start_time"] != -1]
        steps.sort(key=lambda s: s["start_time"])

        if len(steps) < 2:
            dropped.append({"recording_id": rec["recording_id"], "recipe": recipe,
                            "reason": "fewer than 2 performed steps"})
            continue
        out.append(
            Execution(
                recording_id=rec["recording_id"],
                recipe=recipe,
                person_id=rec["person_id"],
                environment=rec["environment"],
                step_ids=[s["step_id"] for s in steps],
                durations={s["step_id"]: round(s["end_time"] - s["start_time"], 2) for s in steps},
                has_errors={s["step_id"]: bool(s.get("has_errors")) for s in steps},
            )
        )

    if len(dropped) > allow_dropped:
        raise UnprojectedStepError(
            f"{len(dropped)} recording(s) dropped for unprojected steps, "
            f"allow_dropped={allow_dropped}. The projection has regressed or the "
            f"pinned annotation release changed. Details: {dropped[:5]}"
        )
    load_executions.dropped = dropped
    return sorted(out, key=lambda e: e.recording_id)


def build_decisions(ex: Execution, proto: Protocol, min_candidates: int = 2) -> list[Decision]:
    """Every point in an execution where >= min_candidates distinct steps remain.

    Candidates are ALL remaining steps, not just task-graph-ready ones. Filtering
    to ready-only would bake the protocol into the harness and shrink the task;
    instead the protocol is given to every arm in the prompt, so all arms see the
    same constraints and the skill is the only thing that varies.

    Repeated steps: a step performed twice stays in the candidate set until its
    last occurrence, but candidates are DEDUPLICATED so the ranking target is
    unambiguous.
    """
    decisions = []
    done: list[int] = []
    remaining = list(ex.step_ids)
    for i, sid in enumerate(ex.step_ids):
        cands = sorted(set(remaining))
        if len(cands) >= min_candidates:
            decisions.append(
                Decision(
                    recording_id=ex.recording_id,
                    recipe=ex.recipe,
                    person_id=ex.person_id,
                    index=i,
                    done=list(done),
                    candidates=cands,
                    true_next=sid,
                    ready=proto.ready(set(done), set(cands)),
                )
            )
        done.append(sid)
        remaining.remove(sid)
    return decisions


def load_all(min_candidates: int = 2):
    protocols = load_protocols()
    executions = load_executions(protocols)
    decisions = []
    for ex in executions:
        decisions.extend(build_decisions(ex, protocols[ex.recipe], min_candidates))
    return protocols, executions, decisions


if __name__ == "__main__":
    import statistics

    protocols, executions, decisions = load_all()
    persons = sorted({e.person_id for e in executions})
    print(f"annotations commit : {ANNOTATIONS_COMMIT}")
    print(f"recipes (protocols): {len(protocols)}")
    print(f"executions         : {len(executions)}")
    print(f"participants       : {len(persons)} -> {persons}")
    print(f"error sessions     : {sum(e.is_error_session for e in executions)}")
    print(f"decision points    : {len(decisions)}")
    print(f"mean candidates    : {statistics.mean(len(d.candidates) for d in decisions):.2f}")
    print(f"mean ready         : {statistics.mean(len(d.ready) for d in decisions):.2f}")
    blank = sum(1 for p in protocols.values() for t in p.steps.values() if not t.strip())
    print(f"blank step texts   : {blank}   (must be 0)")
    cov = [len(set(e.step_ids) - set(protocols[e.recipe].steps)) for e in executions]
    print(f"unmapped step ids  : {sum(cov)}   (must be 0)")
