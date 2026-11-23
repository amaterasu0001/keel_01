"""Strengthened-test labels (SWE-ABS, Yu et al., ICML 2026) for records.

SWE-ABS publishes stronger test patches for SWE-bench Verified on Hugging Face
(`OpenAgentLab/SWE-Bench_Verified_ABS`, Apache-2.0) and ships its own evaluation harness
(github.com/OpenAgentEval/SWE-ABS). Their harness writes, per run,

    swe_abs_res/eval_agent/<run_id>/final_results.json
    {"<instance_id>": {"pass": ["<model key>", ...], "fail": ["<model key>", ...]}}

where a model key is the submission folder name (e.g. 20250807_openhands_gpt5). Run it on the
submission folders you used for the records (see docs/thesis-gap1.md), then merge the result here.

Input layout. The harness reads `<vaild_model_path>/<model>/logs/<instance_id>/{report.json,patch.diff}`
(or `<model>/<instance_id>/...`) and only evaluates instances whose original report says resolved;
the rest go to `init_fail`. `keel eval fetch` downloads exactly this layout, so its output folder is
the `--vaild_model_path`. `abs_command` / `run_sweabs` build and run the call; `final_results_path`
finds the output.

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
    """{instance_id: {model_key: wrong}} from a SWE-ABS final_results.json.

    Only "pass" / "fail" become labels. Models in "init_fail" (not resolved originally, or files
    missing) and "error" get no ABS label here; see `final_results_problems`.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, bool]] = {}
    for iid, r in data.items():
        m = out.setdefault(iid, {})
        for key in r.get("pass", []):
            m[key] = False
        for key in r.get("fail", []):
            m[key] = True
    return out


def final_results_problems(path: str | Path) -> dict[str, dict[str, list[str]]]:
    """{model_key: {"init_fail": [instances], "error": [instances]}} from a final_results.json."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, list[str]]] = {}
    for iid, r in data.items():
        for key in r.get("init_fail", []):
            out.setdefault(key, {"init_fail": [], "error": []})["init_fail"].append(iid)
        for e in r.get("error", []):
            key = e[0] if isinstance(e, (list, tuple)) and e else str(e)
            out.setdefault(key, {"init_fail": [], "error": []})["error"].append(iid)
    return out


def final_results_path(sweabs_dir: str | Path, run_id: str) -> Path:
    """Where the SWE-ABS harness writes its output (relative to its swe-bench/ working directory)."""
    return Path(sweabs_dir) / "swe-bench" / "swe_abs_res" / "eval_agent" / run_id / "final_results.json"


def resolved_instances(model_dir: str | Path) -> list[str]:
    """Instances a submission resolved under the original tests (the only ones SWE-ABS evaluates)."""
    p = Path(model_dir) / "results" / "results.json"
    if p.exists():
        return sorted(json.loads(p.read_text(encoding="utf-8")).get("resolved", []))
    out = []
    for rep in sorted(Path(model_dir).glob("logs/*/report.json")):
        for iid, r in json.loads(rep.read_text(encoding="utf-8")).items():
            if r.get("resolved"):
                out.append(iid)
    return sorted(out)


def ready_instances(model_dir: str | Path) -> list[str]:
    """Resolved instances that also have logs/<id>/{report.json,patch.diff} on disk: what SWE-ABS will evaluate."""
    d = Path(model_dir)
    return [i for i in resolved_instances(d)
            if (d / "logs" / i / "report.json").exists() and (d / "logs" / i / "patch.diff").exists()]


def check_layout(models_root: str | Path, models: list[str]) -> dict[str, int]:
    """Number of instances per model that have both report.json and patch.diff (what SWE-ABS needs)."""
    out = {}
    for m in models:
        d = Path(models_root) / m
        out[m] = sum(1 for p in d.glob("logs/*") if (p / "report.json").exists() and (p / "patch.diff").exists())
    return out


def abs_command(models_root: str | Path, models: list[str], run_id: str, instance_ids: list[str] | None = None,
                workers: int = 8, timeout: int = 1800, python: str = "python",
                predictions: str = ABS_DATASET, rerun: bool = False) -> list[str]:
    """The SWE-ABS harness call (run it from <SWE-ABS>/swe-bench). Flag spelling `vaild_` is theirs."""
    cmd = [python, "-m", "swebench.runtest.run_evaluation_test",
           "--predictions_test_path", predictions,
           "--vaild_model_path", str(Path(models_root).resolve()),
           "--vaild_model_name", ",".join(models),
           "--run_id", run_id, "--max_workers", str(workers), "--timeout", str(timeout)]
    if instance_ids:
        cmd += ["--instance_ids", ",".join(instance_ids)]
    if rerun:
        cmd += ["--re_run_eval", "True"]
    return cmd


def run_sweabs(sweabs_dir: str | Path, cmd: list[str], log=print) -> int:
    """Run the harness in <SWE-ABS>/swe-bench (needs Linux x86, Docker and the swe-abs environment)."""
    import subprocess
    cwd = Path(sweabs_dir) / "swe-bench"
    if not (cwd / "swebench" / "runtest").exists():
        raise FileNotFoundError(f"{cwd} does not look like a SWE-ABS checkout (no swebench/runtest)")
    log("running: " + " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd).returncode


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
