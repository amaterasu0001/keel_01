"""Bug records: every candidate's features plus its correctness labels.

One JSON object per line:
{"task_id": ..., "dataset": ..., "simulated": false,
 "candidates": [{"candidate_id", "generator", "features": {...}, "gates": [...],
                 "wrong": bool, "cost_usd": float, "diff_hash": str,
                 "labels": {"orig": bool, "abs": bool | null, "human": bool | null, "true": bool}}]}

`wrong` is the primary oracle label (the held-out tests) and is what older records carry. The
optional `labels` dict holds one label per oracle, True = wrong, None = not labelled:

  orig   the benchmark's own held-out tests (same as `wrong`)
  abs    strengthened tests (SWE-ABS)
  human  a human audit
  true   ground truth, only in simulated records

Pseudo-oracles: "best" = human if present, else abs, else orig.
"""

from __future__ import annotations

import json
from pathlib import Path


def load(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save(path: str | Path, records: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def append(path: str | Path, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


ORACLES = ("orig", "abs", "human", "true", "best")


def label(c: dict, oracle: str = "orig") -> bool | None:
    """The candidate's label under an oracle (True = wrong), or None if that oracle has no label."""
    labs = c.get("labels") or {}
    if oracle in ("orig", "wrong"):
        v = labs.get("orig")
        return bool(c["wrong"]) if v is None else bool(v)
    if oracle == "best":
        for o in ("human", "abs"):
            if labs.get(o) is not None:
                return bool(labs[o])
        return label(c, "orig")
    if oracle not in ORACLES:
        raise ValueError(f"unknown oracle {oracle!r}; use one of {ORACLES}")
    v = labs.get(oracle)
    return None if v is None else bool(v)


def set_label(c: dict, oracle: str, value: bool | None) -> None:
    labs = c.setdefault("labels", {})
    if "orig" not in labs:
        labs["orig"] = bool(c["wrong"])
    labs[oracle] = None if value is None else bool(value)


def has_oracle(records: list[dict], oracle: str) -> bool:
    """True if every candidate has a label under this oracle."""
    return all(label(c, oracle) is not None for r in records for c in r["candidates"])


def restrict(records: list[dict], generators: set[str]) -> list[dict]:
    """Keep only candidates from the given generators (bugs with none left are dropped)."""
    out = []
    for r in records:
        cs = [c for c in r["candidates"] if c["generator"] in generators]
        if cs:
            out.append({**r, "candidates": cs})
    return out
