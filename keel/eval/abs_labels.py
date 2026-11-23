"""Strengthened-test labels (SWE-ABS, Yu et al., ICML 2026) for records.

SWE-ABS publishes stronger test patches for SWE-bench Verified on Hugging Face
(`OpenAgentLab/SWE-Bench_Verified_ABS`, Apache-2.0) and ships its own evaluation harness
(github.com/OpenAgentEval/SWE-ABS). Their harness writes, per run,

    swe_abs_res/eval_agent/<run_id>/final_results.json
    {"<instance_id>": {"pass": ["<model key>", ...], "fail": ["<model key>", ...]}}

where a model key is the submission folder name (e.g. 20250807_openhands_gpt5). Run it on the
submission folders you used for the records (see docs/thesis-gap1.md), then merge the result here.

A second route runs the official `swebench` harness with the ABS dataset swapped in. The dataset
card says only `test_patch` changed (FAIL_TO_PASS / PASS_TO_PASS lists did not), so new test
functions may not be counted by the official harness; prefer the SWE-ABS harness and use this
route only as a cross-check.
"""

from __future__ import annotations

import json
from pathlib import Path

from .records import label as rec_label
from .records import set_label

ABS_DATASET = "OpenAgentLab/SWE-Bench_Verified_ABS"


def read_final_results(path: str | Path) -> dict[str, dict[str, bool]]:
    """{instance_id: {model_key: wrong}} from a SWE-ABS final_results.json."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, bool]] = {}
    for iid, r in data.items():
        m = out.setdefault(iid, {})
        for key in r.get("pass", []):
            m[key] = False
        for key in r.get("fail", []):
            m[key] = True
    return out


def read_harness_reports(workdir: str | Path, run_id: str) -> dict[str, bool]:
    """{instance_id: wrong} from official-harness reports (logs/run_evaluation/<run>/<model>/<iid>/report.json)."""
    out = {}
    for rep in Path(workdir).glob(f"logs/run_evaluation/{run_id}/*/*/report.json"):
        for iid, r in json.loads(rep.read_text(encoding="utf-8")).items():
            out[iid] = not bool(r.get("resolved"))
    return out


def merge(records: list[dict], labels: dict[str, dict[str, bool]] | dict[str, bool],
          key_map: dict[str, str] | None = None, monotone: bool = True, log=print) -> dict[str, int]:
    """Set labels["abs"] on every candidate found in `labels`.

    labels: {instance: {model_key: wrong}} or {instance: wrong} (then it applies to every candidate).
    key_map: generator name -> SWE-ABS model key, when they differ.
    monotone: strengthened tests only ADD tests, so a patch that fails the original tests is kept
      wrong even if the ABS run says pass (that would be an environment difference, counted).
    Gated candidates (no patch, did not apply) are wrong under every oracle.
    """
    key_map = key_map or {}
    stats = {"labelled": 0, "missing": 0, "flipped_to_wrong": 0, "abs_pass_but_orig_fail": 0}
    for r in records:
        per = labels.get(r["task_id"])
        for c in r["candidates"]:
            if c["gates"]:
                set_label(c, "abs", True)
                stats["labelled"] += 1
                continue
            if per is None:
                stats["missing"] += 1
                continue
            if isinstance(per, dict):
                key = key_map.get(c["generator"], c["generator"])
                if key not in per:
                    stats["missing"] += 1
                    continue
                v = per[key]
            else:
                v = bool(per)
            orig = rec_label(c, "orig")
            if orig and not v:
                stats["abs_pass_but_orig_fail"] += 1
                if monotone:
                    v = True
            if v and not orig:
                stats["flipped_to_wrong"] += 1
            set_label(c, "abs", v)
            stats["labelled"] += 1
    log(f"abs labels: {stats}")
    return stats


def official_harness_command(predictions: str | Path, run_id: str, workers: int = 4) -> list[str]:
    """Cross-check route: the official harness with the ABS dataset (see module docstring caveat)."""
    import sys
    return [sys.executable, "-m", "swebench.harness.run_evaluation", "--dataset_name", ABS_DATASET,
            "--predictions_path", str(predictions), "--max_workers", str(workers), "--run_id", run_id,
            "--cache_level", "env"]
