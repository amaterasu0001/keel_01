"""Human audit: export the patches that need a human verdict, import the verdicts, measure agreement.

Protocol (docs/thesis-gap1.md): only audit-pool patches that pass an oracle need a human label.
Raters see the issue, the agent's patch and the reference (gold) patch, but not the oracle results,
the score or the agent's name. Two raters label each item correct / wrong / unsure; disagreements
go to a third person (`final` column). Unsure counts as wrong by default (the safe side).
"""

from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

from .records import set_label
from .thesis import audit_requests

VERDICTS = {"correct": False, "wrong": True, "unsure": None}
SHEET_FIELDS = ["audit_id", "rater_1", "rater_2", "final", "notes"]


def _audit_id(task_id: str, candidate_id: str) -> str:
    return "A" + hashlib.sha256(f"{task_id}|{candidate_id}".encode()).hexdigest()[:8]


def export(records: list[dict], instances: dict[str, dict], out_dir: str | Path, audit_frac: float = 0.25,
           oracles: tuple[str, ...] = ("orig", "abs"), double_frac: float = 0.3, seed: int = 0,
           salt: str = "keel-audit") -> dict:
    """Write out_dir/items/<audit_id>.md, out_dir/sheet.csv (blind) and out_dir/key.csv (private).

    double_frac of the items are marked for a second rater (rater_2), for Cohen's kappa.
    """
    out = Path(out_dir)
    (out / "items").mkdir(parents=True, exist_ok=True)
    reqs = audit_requests(records, audit_frac, oracles, salt)
    cands = {(r["task_id"], c["candidate_id"]): c for r in records for c in r["candidates"]}
    rng = random.Random(seed)
    order = list(range(len(reqs)))
    rng.shuffle(order)  # blind ordering: not grouped by agent or score
    double = set(rng.sample(order, int(round(double_frac * len(order))))) if order else set()
    sheet, key = [], []
    for i in order:
        q = reqs[i]
        c = cands[(q["task_id"], q["candidate_id"])]
        inst = instances.get(q["task_id"], {})
        aid = _audit_id(q["task_id"], q["candidate_id"])
        (out / "items" / f"{aid}.md").write_text(
            f"# {aid}\n\nRepository: `{inst.get('repo', '?')}`\n\n## Issue\n\n{inst.get('problem_statement', '')}\n\n"
            f"## Patch under review\n\n```diff\n{c.get('diff', '(diff not stored in the record)')}\n```\n\n"
            f"## Reference patch (developer's fix)\n\n```diff\n{inst.get('patch', '(not available)')}\n```\n\n"
            "## Checklist\n\n- Does the patch resolve what the issue asks for, in all the cases it describes?\n"
            "- Does it break or change other behaviour that the reference patch keeps?\n"
            "- If it differs from the reference, is the difference acceptable to a maintainer?\n\n"
            "Verdict: correct / wrong / unsure\n", encoding="utf-8")
        sheet.append({"audit_id": aid, "rater_1": "", "rater_2": "" if i in double else "-", "final": "",
                      "notes": ""})
        key.append({"audit_id": aid, "task_id": q["task_id"], "candidate_id": q["candidate_id"],
                    "generator": q["generator"], "passes": "+".join(q["passes"])})
    _write_csv(out / "sheet.csv", SHEET_FIELDS, sheet)
    _write_csv(out / "key.csv", ["audit_id", "task_id", "candidate_id", "generator", "passes"], key)
    return {"items": len(sheet), "double_rated": sum(1 for s in sheet if s["rater_2"] != "-")}


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _norm(v: str) -> str | None:
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


def import_verdicts(records: list[dict], audit_dir: str | Path, unsure_as_wrong: bool = True) -> dict:
    """Read sheet.csv + key.csv and set labels["human"]. Returns counts and Cohen's kappa."""
    d = Path(audit_dir)
    key = {r["audit_id"]: r for r in _read_csv(d / "key.csv")}
    cands = {(r["task_id"], c["candidate_id"]): c for r in records for c in r["candidates"]}
    pairs_a, pairs_b = [], []
    stats = {"labelled": 0, "unlabelled": 0, "disagreements_unresolved": 0, "unsure": 0}
    for row in _read_csv(d / "sheet.csv"):
        k = key.get(row["audit_id"])
        if not k:
            continue
        r1, r2, fin = _norm(row.get("rater_1")), _norm(row.get("rater_2")), _norm(row.get("final"))
        if r1 and r2:
            pairs_a.append(r1)
            pairs_b.append(r2)
        verdict = fin or (r1 if (r2 is None or r1 == r2) else None)
        if r1 and r2 and r1 != r2 and not fin:
            stats["disagreements_unresolved"] += 1
            verdict = "unsure"
        if verdict is None:
            stats["unlabelled"] += 1
            continue
        if verdict == "unsure":
            stats["unsure"] += 1
        value = VERDICTS[verdict]
        if value is None:
            value = True if unsure_as_wrong else None
        c = cands.get((k["task_id"], k["candidate_id"]))
        if c is not None and value is not None:
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
