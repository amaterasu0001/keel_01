"""Oracle 3, the human audit: export items, collect two independent verdicts, adjudicate, import.

Protocol (docs/thesis-gap1.md): only audit-pool patches that pass an oracle need a human label.
Raters see the issue, the agent's patch and the reference (gold) patch, but not the oracle results,
the score or the agent's name, and not each other's verdicts.

Files in the audit folder:

  items/<audit_id>.md   what a rater reads (issue, patch, reference patch, checklist)
  rater1.csv            every item: audit_id, verdict, notes          -> first rater
  rater2.csv            the double-rated subset (default 30%)          -> second rater, separately
  adjudication.csv      written by `audit_status` for disagreements    -> third person fills `final`
  key.csv               audit_id -> task, candidate, agent             -> keep away from raters

Verdicts: correct / wrong / unsure (also c / w / u). Unsure counts as wrong by default.
An older single `sheet.csv` (rater_1, rater_2, final columns) is still read if present.
"""

from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

from .records import set_label
from .thesis import audit_requests

VERDICTS = {"correct": False, "wrong": True, "unsure": None}
RATER_FIELDS = ["audit_id", "verdict", "notes"]
ADJ_FIELDS = ["audit_id", "rater_1", "rater_2", "final", "notes"]
SHEET_FIELDS = ["audit_id", "rater_1", "rater_2", "final", "notes"]  # legacy single sheet


def _audit_id(task_id: str, candidate_id: str) -> str:
    return "A" + hashlib.sha256(f"{task_id}|{candidate_id}".encode()).hexdigest()[:8]


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:  # -sig: tolerate Excel's BOM
        return list(csv.DictReader(f))


def _item_markdown(aid: str, inst: dict, diff: str) -> str:
    return (f"# {aid}\n\nRepository: `{inst.get('repo', '?')}`\n\n## Issue\n\n{inst.get('problem_statement', '')}\n\n"
            f"## Patch under review\n\n```diff\n{diff or '(diff not stored in the record)'}\n```\n\n"
            f"## Reference patch (developer's fix)\n\n```diff\n{inst.get('patch', '(not available)')}\n```\n\n"
            "## Checklist\n\n"
            "- Does the patch resolve what the issue asks for, in all the cases it describes?\n"
            "- Does it break or change other behaviour that the reference patch keeps?\n"
            "- If it differs from the reference, is the difference acceptable to a maintainer?\n"
            "- Judge the code, not whether tests pass (that information is hidden on purpose).\n\n"
            "Verdict: correct / wrong / unsure\n")


def export(records: list[dict], instances: dict[str, dict], out_dir: str | Path, audit_frac: float = 0.25,
           oracles: tuple[str, ...] = ("orig", "abs"), double_frac: float = 0.3, seed: int = 0,
           salt: str = "keel-audit") -> dict:
    """Write the audit folder (see module docstring). Re-running keeps verdicts already entered."""
    out = Path(out_dir)
    (out / "items").mkdir(parents=True, exist_ok=True)
    reqs = audit_requests(records, audit_frac, oracles, salt)
    cands = {(r["task_id"], c["candidate_id"]): c for r in records for c in r["candidates"]}
    rng = random.Random(seed)
    order = list(range(len(reqs)))
    rng.shuffle(order)  # blind ordering: not grouped by agent or score
    double = set(rng.sample(order, int(round(double_frac * len(order))))) if order else set()
    old1 = {r["audit_id"]: r for r in _read_csv(out / "rater1.csv")}
    old2 = {r["audit_id"]: r for r in _read_csv(out / "rater2.csv")}
    r1, r2, key = [], [], []
    for i in order:
        q = reqs[i]
        c = cands[(q["task_id"], q["candidate_id"])]
        aid = _audit_id(q["task_id"], q["candidate_id"])
        md = _item_markdown(aid, instances.get(q["task_id"], {}), c.get("diff", ""))
        (out / "items" / f"{aid}.md").write_text(md, encoding="utf-8")
        r1.append(old1.get(aid) or {"audit_id": aid, "verdict": "", "notes": ""})
        if i in double:
            r2.append(old2.get(aid) or {"audit_id": aid, "verdict": "", "notes": ""})
        key.append({"audit_id": aid, "task_id": q["task_id"], "candidate_id": q["candidate_id"],
                    "generator": q["generator"], "passes": "+".join(q["passes"])})
    _write_csv(out / "rater1.csv", RATER_FIELDS, r1)
    _write_csv(out / "rater2.csv", RATER_FIELDS, r2)
    _write_csv(out / "key.csv", ["audit_id", "task_id", "candidate_id", "generator", "passes"], key)
    return {"items": len(r1), "double_rated": len(r2)}


def _norm(v: str | None) -> str | None:
    v = (v or "").strip().lower()
    if v in ("", "-"):
        return None
    aliases = {"c": "correct", "ok": "correct", "w": "wrong", "x": "wrong", "u": "unsure", "?": "unsure"}
    v = aliases.get(v, v)
    if v not in VERDICTS:
        raise ValueError(f"unknown verdict {v!r}; use correct / wrong / unsure")
    return v


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two raters over the same items (categorical labels)."""
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def _verdicts(d: Path) -> tuple[dict, dict, dict]:
    """(rater1, rater2, final) verdicts by audit id, from the new files or a legacy sheet.csv."""
    v1 = {r["audit_id"]: _norm(r.get("verdict")) for r in _read_csv(d / "rater1.csv")}
    v2 = {r["audit_id"]: _norm(r.get("verdict")) for r in _read_csv(d / "rater2.csv")}
    fin = {r["audit_id"]: _norm(r.get("final")) for r in _read_csv(d / "adjudication.csv")}
    if not v1 and (d / "sheet.csv").exists():
        for r in _read_csv(d / "sheet.csv"):
            v1[r["audit_id"]] = _norm(r.get("rater_1"))
            if (r.get("rater_2") or "").strip() != "-":
                v2[r["audit_id"]] = _norm(r.get("rater_2"))
            fin[r["audit_id"]] = _norm(r.get("final"))
    return v1, v2, fin


def audit_status(audit_dir: str | Path, write_adjudication: bool = True) -> dict:
    """Progress, agreement, and (optionally) adjudication.csv for the items the two raters disagree on."""
    d = Path(audit_dir)
    v1, v2, fin = _verdicts(d)
    both = [a for a in v2 if v1.get(a) and v2.get(a)]
    dis = [a for a in both if v1[a] != v2[a]]
    st = {"items": len(v1), "rater1_done": sum(1 for v in v1.values() if v), "rater2_items": len(v2),
          "rater2_done": sum(1 for v in v2.values() if v), "double_rated_done": len(both),
          "disagreements": len(dis), "adjudicated": sum(1 for a in dis if fin.get(a)),
          "kappa": cohen_kappa([v1[a] for a in both], [v2[a] for a in both]) if both else None}
    if write_adjudication and dis:
        old = {r["audit_id"]: r for r in _read_csv(d / "adjudication.csv")}
        rows = [{"audit_id": a, "rater_1": v1[a], "rater_2": v2[a],
                 "final": (old.get(a) or {}).get("final", ""), "notes": (old.get(a) or {}).get("notes", "")}
                for a in sorted(dis)]
        _write_csv(d / "adjudication.csv", ADJ_FIELDS, rows)
    st["ready"] = (st["rater1_done"] == st["items"] and st["rater2_done"] == st["rater2_items"]
                   and st["adjudicated"] == st["disagreements"])
    return st


def import_verdicts(records: list[dict], audit_dir: str | Path, unsure_as_wrong: bool = True) -> dict:
    """Set labels["human"] from the audit folder. Returns counts and Cohen's kappa."""
    d = Path(audit_dir)
    key = {r["audit_id"]: r for r in _read_csv(d / "key.csv")}
    cands = {(r["task_id"], c["candidate_id"]): c for r in records for c in r["candidates"]}
    v1, v2, fin = _verdicts(d)
    pairs_a, pairs_b = [], []
    stats = {"labelled": 0, "unlabelled": 0, "disagreements_unresolved": 0, "unsure": 0, "not_in_records": 0}
    for aid, k in key.items():
        c = cands.get((k["task_id"], k["candidate_id"]))
        if c is None:
            stats["not_in_records"] += 1  # another agent's item: imported into that agent's file
            continue
        a, b, f = v1.get(aid), v2.get(aid), fin.get(aid)
        if a and b:
            pairs_a.append(a)
            pairs_b.append(b)
        if f:
            verdict = f
        elif a and b and a != b:
            stats["disagreements_unresolved"] += 1
            verdict = "unsure"
        else:
            verdict = a if (b is None or a == b) else None
        if verdict is None:
            stats["unlabelled"] += 1
            continue
        if verdict == "unsure":
            stats["unsure"] += 1
        value = VERDICTS[verdict]
        if value is None:
            value = True if unsure_as_wrong else None
        if value is not None:
            set_label(c, "human", value)
            stats["labelled"] += 1
    stats["kappa"] = cohen_kappa(pairs_a, pairs_b) if pairs_a else None
    stats["double_rated"] = len(pairs_a)
    return stats


def fill_human_from_oracle_failures(records: list[dict], oracle: str = "orig") -> int:
    """Patches the oracle rejects (or that are gated) need no human: mark them wrong. Returns count."""
    from .records import label as rec_label
    n = 0
    for r in records:
        for c in r["candidates"]:
            if (c["gates"] or rec_label(c, oracle)) and rec_label(c, "human") is None:
                set_label(c, "human", True)
                n += 1
    return n
