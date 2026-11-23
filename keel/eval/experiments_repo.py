"""Records from patches that agents have already published on the SWE-bench leaderboard.

The `swe-bench/experiments` repository keeps one folder per submission, e.g.
`evaluation/verified/20240620_sweagent_claude3.5sonnet/`, usually with:

  all_preds.jsonl          one {"instance_id", "model_patch", "model_name_or_path"} per line
                           (some submissions use preds.json: a list, or a dict keyed by instance id)
  results/results.json     {"resolved": [...], "no_apply": [...], ...} from the official harness
  logs/<instance>/report.json   per-instance harness report (fallback for the labels)
  trajs/                   agent trajectories (sometimes only in a separate download)

No agent is run and no harness is needed for the original label: `orig` = not resolved by the
official SWE-bench tests. Strengthened-test (`abs`) and human labels are merged in later
(abs_labels.py, audit.py).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..diffutil import normalized_diff
from . import signals_ext as sx


def _read_preds(folder: Path) -> dict[str, str]:
    for name in ("all_preds.jsonl", "predictions.jsonl", "preds.jsonl"):
        p = folder / name
        if p.exists():
            out = {}
            for ln in p.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    row = json.loads(ln)
                    out[row["instance_id"]] = row.get("model_patch") or ""
            return out
    for name in ("preds.json", "all_preds.json"):
        p = folder / name
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {k: (v.get("model_patch") if isinstance(v, dict) else v) or "" for k, v in data.items()}
            return {row["instance_id"]: row.get("model_patch") or "" for row in data}
    raise FileNotFoundError(f"no predictions file (all_preds.jsonl / preds.json) in {folder}")


def _read_results(folder: Path) -> tuple[set[str] | None, set[str]]:
    """(resolved ids or None if unknown, ids whose patch did not apply)."""
    p = folder / "results" / "results.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data.get("resolved"), list):
            return set(data["resolved"]), set(data.get("no_apply", []))
        if isinstance(data.get("resolved_ids"), list):
            return set(data["resolved_ids"]), set()
    logs = folder / "logs"
    if logs.exists():
        resolved, no_apply = set(), set()
        for rep in logs.glob("*/report.json"):
            body = json.loads(rep.read_text(encoding="utf-8"))
            for iid, r in body.items():
                if r.get("resolved"):
                    resolved.add(iid)
                if r.get("patch_successfully_applied") is False:
                    no_apply.add(iid)
        return resolved, no_apply
    return None, set()


def load_submission(folder: str | Path, instances: dict[str, dict], name: str | None = None,
                    judge=None, with_traj: bool = True, log=print) -> list[dict]:
    """One record per instance (one candidate: this submission's patch).

    instances: SWE-bench rows keyed by instance id (problem_statement is used for issue features).
    Instances the submission did not attempt get an empty patch, gated as "no_patch".
    """
    folder = Path(folder)
    name = name or folder.name
    preds = _read_preds(folder)
    resolved, no_apply = _read_results(folder)
    if resolved is None:
        raise FileNotFoundError(f"{folder}: no results/results.json or logs/*/report.json to label the patches")
    traj_dir = folder / "trajs"
    records = []
    for iid, inst in sorted(instances.items()):
        diff = preds.get(iid, "") or ""
        gates = []
        if not diff.strip():
            gates.append("no_patch")
        elif iid in no_apply:
            gates.append("patch_did_not_apply")
        feats = {**sx.diff_features(diff), **sx.issue_features(inst.get("problem_statement", ""), diff)}
        feats.update(sx.trajectory_features(sx.load_trajectory(traj_dir, iid) if with_traj else None))
        feats.update(sx.judge_features(judge, inst.get("problem_statement", ""), diff) if diff.strip() and judge
                     else {"has_judge": 0.0, "judge_correct_prob": 0.5})
        wrong = iid not in resolved
        records.append({
            "task_id": iid, "dataset": "swebench-verified", "simulated": False, "submission": name,
            "candidates": [{"candidate_id": f"{name}-0", "generator": name, "features": feats, "gates": gates,
                            "wrong": wrong, "labels": {"orig": wrong, "abs": None, "human": None},
                            "cost_usd": 0.0, "diff": diff,
                            "diff_hash": hashlib.sha256(normalized_diff(diff).encode()).hexdigest()[:16]}],
        })
    n_res = sum(1 for r in records if not r["candidates"][0]["wrong"])
    log(f"{name}: {len(records)} instances, {len(preds)} predictions, {n_res} resolved by the official tests")
    return records


def load_instances_file(path: str | Path) -> dict[str, dict]:
    """SWE-bench rows from a local .json (list) or .jsonl file, keyed by instance id."""
    text = Path(path).read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(ln) for ln in text.splitlines()
                                                                     if ln.strip()]
    return {r["instance_id"]: r for r in rows}
