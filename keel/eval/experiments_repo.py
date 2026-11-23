"""Records from patches that agents have already published on the SWE-bench leaderboard.

The `swe-bench/experiments` repository keeps one folder per submission, e.g.
`evaluation/verified/20250807_openhands_gpt5/`. In the repository itself the folder holds only
metadata.yaml, README.md and results/results.json ({"resolved": [...], "no_generation": [...], ...}).
The patches, logs and trajectories live in the submitter's own repository or a public S3 bucket and
are fetched (no credentials needed) with, from the experiments checkout:

  python -m analysis.download_logs evaluation/verified/<submission>

which adds, per submission:

  all_preds.jsonl               one {"instance_id", "model_patch", "model_name_or_path"} per line
  logs/<instance>/patch.diff    the patch that was evaluated (used if all_preds.jsonl is absent)
  logs/<instance>/report.json   per-instance harness report (fallback for the labels)
  trajs/<instance>.*            agent trajectories

If the download puts all_preds.jsonl somewhere else, pass its path as `preds_path`.

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


def _jsonl_preds(p: Path) -> dict[str, str]:
    out = {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            row = json.loads(ln)
            out[row["instance_id"]] = row.get("model_patch") or ""
    return out


def _json_preds(p: Path) -> dict[str, str]:
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {k: (v.get("model_patch") if isinstance(v, dict) else v) or "" for k, v in data.items()}
    return {row["instance_id"]: row.get("model_patch") or "" for row in data}


def _read_preds(folder: Path, preds_path: Path | None = None) -> dict[str, str]:
    """Patches by instance: an explicit file, else all_preds.jsonl / preds.json, else logs/*/patch.diff."""
    if preds_path is not None:
        return _json_preds(preds_path) if preds_path.suffix == ".json" else _jsonl_preds(preds_path)
    for name in ("all_preds.jsonl", "predictions.jsonl", "preds.jsonl"):
        if (folder / name).exists():
            return _jsonl_preds(folder / name)
    for name in ("preds.json", "all_preds.json"):
        if (folder / name).exists():
            return _json_preds(folder / name)
    diffs = sorted(folder.glob("logs/*/patch.diff"))
    if diffs:
        return {d.parent.name: d.read_text(encoding="utf-8", errors="replace") for d in diffs}
    raise FileNotFoundError(f"no predictions in {folder}: run `python -m analysis.download_logs "
                            f"evaluation/verified/{folder.name}` in the experiments checkout, or pass --preds")


def _no_generation(folder: Path) -> set[str]:
    """Instances the agent produced no patch for (results.json "no_generation")."""
    p = folder / "results" / "results.json"
    if p.exists():
        return set(json.loads(p.read_text(encoding="utf-8")).get("no_generation", []))
    return set()


def _read_results(folder: Path) -> tuple[set[str] | None, set[str]]:
    """(resolved ids or None if unknown, ids whose patch did not apply)."""
    p = folder / "results" / "results.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data.get("resolved"), list):
            no_apply = set(data.get("no_apply", []))
            for rep in folder.glob("logs/*/report.json"):  # finer-grained, when downloaded
                for iid, r in json.loads(rep.read_text(encoding="utf-8")).items():
                    if r.get("patch_successfully_applied") is False:
                        no_apply.add(iid)
            return set(data["resolved"]), no_apply
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
                    judge=None, with_traj: bool = True, log=print, preds_path: str | Path | None = None) -> list[dict]:
    """One record per instance (one candidate: this submission's patch).

    instances: SWE-bench rows keyed by instance id (problem_statement is used for issue features).
    Instances the agent produced no patch for (results.json "no_generation") get an empty patch,
    gated as "no_patch" (wrong under every oracle). Instances whose patch was simply not downloaded
    (e.g. a `keel eval fetch --limit` pilot) are left out, so a pilot never invents labels.
    """
    folder = Path(folder)
    name = name or folder.name
    preds = _read_preds(folder, Path(preds_path) if preds_path else None)
    resolved, no_apply = _read_results(folder)
    if resolved is None:
        raise FileNotFoundError(f"{folder}: no results/results.json or logs/*/report.json to label the patches")
    traj_dir = folder / "trajs"
    records = []
    no_gen = _no_generation(folder)
    skipped = 0
    for iid, inst in sorted(instances.items()):
        if iid not in preds and iid not in no_gen:
            skipped += 1
            continue
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
    log(f"{name}: {len(records)} instances, {len(preds)} predictions, {n_res} resolved by the official tests"
        + (f"; {skipped} instances without a downloaded patch left out" if skipped else ""))
    return records


def load_instances_file(path: str | Path) -> dict[str, dict]:
    """SWE-bench rows from a local .json (list) or .jsonl file, keyed by instance id."""
    text = Path(path).read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(ln) for ln in text.splitlines()
                                                                     if ln.strip()]
    return {r["instance_id"]: r for r in rows}
