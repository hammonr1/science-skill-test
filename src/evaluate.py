"""Frozen-evaluator harness for the four arms.

Determinism: temperature=0, fixed seeds for every shuffle/scramble, and an
on-disk response cache keyed by a hash of (model, system, user prompt). Re-runs
hit the cache, so the script is reproducible and resumable, and a crash midway
costs nothing.

Metric: MRR (mean reciprocal rank) of the true next step among all remaining
steps, with top-1 accuracy logged alongside. MRR is the primary because candidate
sets average ~9 and top-1 throws away most of the signal in each ranking; MRR
uses the whole ordering and so needs far fewer API calls for the same power.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from prompts import SYSTEM, build_user_prompt
from skills import mine_residual, render_skill_block

ARMS = ["A", "B", "C", "D"]
ARM_LABEL = {
    "A": "no skill (baseline)",
    "B": "flat advice",
    "C": "cue-conditioned",
    "D": "scrambled cue (control)",
}


class Cache:
    """Content-addressed response cache; makes the whole run resumable."""

    def __init__(self, path="results/cache.sqlite"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self.conn.commit()
        self.lock = threading.Lock()

    def get(self, k):
        with self.lock:
            r = self.conn.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return r[0] if r else None

    def put(self, k, v):
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, v))
            self.conn.commit()


def parse_ranking(text: str, candidates: list[int]) -> list[int]:
    """Parse the model's comma-separated ranking; append anything it omitted.

    Omissions are appended in their original order rather than dropped, so a
    partial answer is scored fairly instead of crashing the run.
    """
    seen, out = set(), []
    for tok in re.findall(r"\d+", text or ""):
        v = int(tok)
        if v in candidates and v not in seen:
            seen.add(v)
            out.append(v)
    out.extend(s for s in candidates if s not in seen)
    return out


def call_model(client, model, system, user, cache, max_retries=5):
    key = hashlib.sha256(f"{model}\x00{system}\x00{user}".encode()).hexdigest()
    hit = cache.get(key)
    if hit is not None:
        return hit, True
    delay = 2.0
    for attempt in range(max_retries):
        try:
            r = client.messages.create(
                model=model,
                max_tokens=200,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = r.content[0].text if r.content else ""
            cache.put(key, text)
            return text, False
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def build_jobs(protocols, executions, decisions, recipes=None, k_skills=8, seed=7,
               max_per_fold=None):
    """One job per (decision, arm). Skills are mined per fold from training
    participants only -- the operator hold-out is asserted here, not assumed."""
    persons = sorted({e.person_id for e in executions})
    jobs = []
    for held in persons:
        train_ex = [e for e in executions if e.person_id != held]
        assert all(e.person_id != held for e in train_ex), "operator leak in authoring set"
        eval_d = [d for d in decisions if d.person_id == held]
        if recipes is not None:
            eval_d = [d for d in eval_d if d.recipe in recipes]
        assert all(d.person_id == held for d in eval_d), "operator leak in eval set"
        if max_per_fold:
            eval_d = eval_d[:max_per_fold]

        skills_by_recipe = {}
        for r in {d.recipe for d in eval_d}:
            skills_by_recipe[r] = mine_residual(train_ex, protocols, r, k=k_skills)

        for d in eval_d:
            sk = skills_by_recipe[d.recipe]
            for arm in ARMS:
                block = render_skill_block(sk, arm, seed=seed)
                jobs.append({
                    "fold": held,
                    "arm": arm,
                    "recipe": d.recipe,
                    "recording_id": d.recording_id,
                    "index": d.index,
                    "n_candidates": len(d.candidates),
                    "candidates": d.candidates,
                    "true_next": d.true_next,
                    "user": build_user_prompt(d, protocols[d.recipe], block),
                })
    return jobs


def run_jobs(jobs, model, workers=8, verbose=True):
    import anthropic

    client = anthropic.Anthropic()
    cache = Cache()
    done = [0]
    lock = threading.Lock()

    def work(job):
        text, cached = call_model(client, model, SYSTEM, job["user"], cache)
        ranking = parse_ranking(text, job["candidates"])
        rank = ranking.index(job["true_next"]) + 1
        with lock:
            done[0] += 1
            if verbose and done[0] % 200 == 0:
                print(f"  {done[0]}/{len(jobs)}", flush=True)
        return {
            "fold": job["fold"], "arm": job["arm"], "recipe": job["recipe"],
            "recording_id": job["recording_id"], "index": job["index"],
            "n_candidates": job["n_candidates"], "rank": rank,
            "rr": 1.0 / rank, "top1": 1.0 if rank == 1 else 0.0, "cached": cached,
            "raw": text.strip()[:120],
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(work, jobs))


def estimate_cost(jobs, model):
    """Rough pre-flight cost estimate (chars/4 heuristic, no cache credit)."""
    price = {
        "claude-haiku-4-5-20251001": (1.0, 5.0),
        "claude-sonnet-5": (3.0, 15.0),
        "claude-opus-5": (15.0, 75.0),
    }.get(model, (3.0, 15.0))
    in_tok = sum(len(j["user"]) + len(SYSTEM) for j in jobs) / 4
    out_tok = sum(j["n_candidates"] * 4 + 10 for j in jobs)
    cost = in_tok / 1e6 * price[0] + out_tok / 1e6 * price[1]
    return {"calls": len(jobs), "est_input_tokens": int(in_tok),
            "est_output_tokens": int(out_tok), "est_usd": round(cost, 2)}
