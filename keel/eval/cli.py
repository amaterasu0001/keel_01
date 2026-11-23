"""`keel eval ...` and `keel calibrate` sub-commands."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from . import records as rec_io


def _log(m: str) -> None:
    print(f"[keel] {m}", file=sys.stderr)


def cmd_simulate(a: argparse.Namespace) -> int:
    from .simulate import simulate
    recs = simulate(a.n, k=a.k, seed=a.seed, eps_fc=a.eps_fc, eps_fw=a.eps_fw, abs_catch=a.abs_catch,
                    human_error=a.human_error)
    rec_io.save(a.out, recs)
    noise = f" (oracle noise: eps_fc={a.eps_fc}, eps_fw={a.eps_fw})" if a.eps_fc or a.eps_fw else ""
    _log(f"wrote {len(recs)} SIMULATED bug records to {a.out}{noise}")
    return 0


def _pipeline(a: argparse.Namespace):
    from ..cli import build_pipeline
    from ..config import KeelConfig
    cfg = KeelConfig(k=a.k)
    return build_pipeline(cfg, a.models.split(","), None if a.no_attack else a.attacker, a.triage, None, None,
                          a.cache, log=_log)


def cmd_quixbugs(a: argparse.Namespace) -> int:
    from .build import build_records
    from .quixbugs import build_all, hidden_label, load_tasks
    work = Path(a.work)
    tasks = load_tasks(work) if (work / "tasks.json").exists() and not a.rebuild else \
        build_all(Path(a.cache_dir), work, a.only.split(",") if a.only else None, log=_log)
    if a.only:
        tasks = [t for t in tasks if t.name in a.only.split(",")]
    pipe = _pipeline(a)
    items = [(t.bug_task(), (lambda root, t=t: hidden_label(t, root))) for t in tasks]
    build_records(items, pipe, a.out, "quixbugs", reports_dir=a.reports, log=_log)
    return 0


def cmd_swebench(a: argparse.Namespace) -> int:
    from . import swebench as sb
    insts = sb.load_instances(a.n, seed=a.seed, dataset=sb.DATASETS[a.dataset])
    cache = Path(a.cache_dir)
    cand = Path(a.candidates)
    if a.stage == "list-runs":
        print(json.dumps(sb.run_ids(cand)))
        return 0
    if a.stage in ("generate", "all"):
        sb.stage_generate(insts, _pipeline(a), cache, cand, log=_log)
    if a.stage in ("harness", "all"):
        extra = ["--cache_level", "none", "--clean", "True"] if a.low_disk else None
        sb.stage_harness(cand, Path(a.workdir), max_workers=a.workers, dataset=sb.DATASETS[a.dataset], extra_args=extra,
                         only=a.run_ids.split(",") if a.run_ids else None, log=_log)
    if a.stage in ("records", "all"):
        sb.stage_records(insts, cand, Path(a.workdir), cache, None, Path(a.out), log=_log,
                         dataset_label=f"swebench-{a.dataset}")
    return 0


def cmd_analyze(a: argparse.Namespace) -> int:
    from .runner import ablations, evaluate, model_comparison, slider_data
    recs = []
    for p in a.records:
        recs += rec_io.load(p)
    simulated = any(r.get("simulated") for r in recs)
    datasets = sorted({r["dataset"] for r in recs})
    _log(f"analysing {len(recs)} bugs from {datasets} ({'SIMULATED' if simulated else 'real'})")
    feats = _features(a.features, recs)
    ev = evaluate(recs, n_splits=a.splits, delta=a.delta, seed=a.seed, features=feats, cal_label=a.cal_label,
                  eval_label=a.eval_label)
    out = {
        "simulated": simulated, "datasets": datasets, "n_bugs": len(recs),
        "n_candidates": sum(len(r["candidates"]) for r in recs),
        "generators": sorted({c["generator"] for r in recs for c in r["candidates"]}),
        "evaluation": ev,
        "slider": slider_data(recs, seed=a.seed),
        "models": model_comparison(recs, n_splits=max(20, a.splits // 4), seed=a.seed),
        "ablations": ablations(recs, n_splits=max(20, a.splits // 4), seed=a.seed),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    _log(f"wrote {a.out}")
    return 0


def cmd_calibrate(a: argparse.Namespace) -> int:
    """Fit the scorer on part of the records and calibrate on the rest, for deployment."""
    from ..calibration import Calibration, fingerprint
    from .runner import fit_scorer, score_and_select
    recs = []
    for p in a.records:
        recs += rec_io.load(p)
    rng = random.Random(a.seed)
    idx = list(range(len(recs)))
    rng.shuffle(idx)
    cut = int(len(idx) * a.fit_fraction)
    fit = [recs[i] for i in idx[:cut]]
    cal = [recs[i] for i in idx[cut:]]
    from .. import conformal
    scorer = fit_scorer(fit, _features(a.features, recs))
    fit_sel = score_and_select(fit, scorer)
    sel = score_and_select(cal, scorer)
    grid = conformal.order_grid(conformal.grid_from_scores([x["score"] for x in fit_sel]),
                                [x["score"] for x in fit_sel], [x["wrong"] for x in fit_sel], len(sel), 0.1)
    from ..cli import _model_id
    from ..generators import build_generator
    models = a.models.split(",")
    names = [build_generator(m, cache_dir=None).name for m in models]
    fp = a.fingerprint or fingerprint({n: _model_id(m) for n, m in zip(names, models)}, a.k,
                                      attack=not a.no_attack, triage=bool(a.triage))
    c = Calibration(scorer=scorer, scores=[s["score"] for s in sel], wrong=[s["wrong"] for s in sel],
                    fingerprint=fp, source=",".join(sorted({r["dataset"] for r in recs})),
                    simulated=any(r.get("simulated") for r in recs), task_ids=[s["task_id"] for s in sel], grid=grid)
    c.save(a.out)
    _log(f"calibration with {c.n} bugs written to {a.out}; at α=10%: CRC threshold {c.threshold(0.1):.4f}, "
         f"LTT threshold {c.threshold(0.1, 'ltt'):.4f}")
    return 0


def _features(choice: str, recs: list[dict]) -> list[str] | None:
    """Feature set for the scorer: Keel's own run features, or the external-agent set."""
    from ..features import FEATURE_NAMES
    from .signals_ext import EXTERNAL_FEATURES
    if choice == "keel":
        return list(FEATURE_NAMES)
    if choice == "external":
        return list(EXTERNAL_FEATURES)
    external = any(r.get("submission") for r in recs)
    return list(EXTERNAL_FEATURES) if external else None


def _load_all(paths: list[str]) -> list[dict]:
    recs = []
    for p in paths:
        recs += rec_io.load(p)
    return recs


def _instances(a: argparse.Namespace) -> dict[str, dict]:
    from .experiments_repo import load_instances_file
    if a.instances:
        return load_instances_file(a.instances)
    from . import swebench as sb
    return {r["instance_id"]: r for r in sb.load_instances(None, dataset=sb.DATASETS[a.dataset])}


def cmd_submission(a: argparse.Namespace) -> int:
    """Records from published leaderboard submissions (swe-bench/experiments folders)."""
    from .experiments_repo import load_submission
    insts = _instances(a)
    if a.limit:
        insts = dict(sorted(insts.items())[:a.limit])
    judge = None
    if a.judge:
        from ..generators import build_generator
        judge = build_generator(a.judge, cache_dir=a.cache)
    out_dir = Path(a.out_dir)
    if a.preds and len(a.folders) != 1:
        raise SystemExit("--preds works with exactly one submission folder")
    for folder in a.folders:
        recs = load_submission(folder, insts, judge=judge, with_traj=not a.no_traj, log=_log, preds_path=a.preds)
        out = out_dir / f"{Path(folder).name}.jsonl"
        rec_io.save(out, recs)
        _log(f"wrote {out}")
    return 0


def cmd_instances(a: argparse.Namespace) -> int:
    from . import swebench as sb
    from .fetch import save_instances
    n = save_instances(sb.load_instances(None, dataset=sb.DATASETS[a.dataset]), a.out)
    _log(f"wrote {n} SWE-bench {a.dataset} instances to {a.out}")
    return 0


def cmd_fetch(a: argparse.Namespace) -> int:
    """Oracle 1 input: download published submissions (patches, harness reports, results.json)."""
    from .fetch import fetch_submission
    ids = None
    if a.instances:
        from .experiments_repo import load_instances_file
        ids = sorted(load_instances_file(a.instances))
        if a.limit:
            ids = ids[:a.limit]
        ids = set(ids)
    elif a.limit:
        raise SystemExit("--limit needs --instances (the local SWE-bench rows file from `keel eval instances`)")
    bad = 0
    for name in a.submissions:
        s = fetch_submission(name, a.out, a.split, trajs=a.trajs, instances=ids, workers=a.workers, log=_log)
        bad += s["readme_resolved"] is not None and s["readme_resolved"] != s["resolved"]
    _log(f"done. Next: keel eval submission {a.out}/<submission> --instances <rows> --out-dir results/agents")
    return 1 if bad else 0


def cmd_abs_run(a: argparse.Namespace) -> int:
    """Oracle 2: build (and run) the SWE-ABS harness call for downloaded submissions."""
    from . import abs_labels as ab
    models = a.models.split(",")
    layout = ab.check_layout(a.models_root, models)
    for m, n in layout.items():
        _log(f"{m}: {n} instances with report.json + patch.diff")
    if not all(layout.values()):
        raise SystemExit("some models have no logs/<instance>/{report.json,patch.diff}; run `keel eval fetch` first")
    ids = a.instance_ids.split(",") if a.instance_ids else None
    if a.pilot:
        ids = ab.ready_instances(Path(a.models_root) / models[0])[:a.pilot]
        _log(f"pilot: {len(ids)} downloaded instances resolved by {models[0]}")
    if a.docker:
        return _abs_run_docker(a, models, ids)
    cmd = ab.abs_command(a.models_root, models, a.run_id, ids, a.workers, a.timeout, a.python, rerun=a.rerun)
    print(" ".join(cmd))
    if a.dry_run or not a.sweabs_dir:
        _log("not run (pass --sweabs-dir to run it; use it from <SWE-ABS>/swe-bench in the swe-abs environment)")
        return 0
    rc = ab.run_sweabs(a.sweabs_dir, cmd, log=_log)
    out = ab.final_results_path(a.sweabs_dir, a.run_id)
    _log(f"harness exit code {rc}; results: {out}" + ("" if out.exists() else " (missing!)"))
    return rc


def _abs_run_docker(a: argparse.Namespace, models: list[str], ids: list[str] | None) -> int:
    from . import abs_docker as dk
    cmd = dk.run_command(a.models_root, models, a.run_id, a.results_dir, ids, a.workers, a.timeout, a.image,
                         a.rerun, host_uid_gid=dk.host_ids(), name=f"keel-oracle2-{a.run_id}")
    print(" ".join(cmd))
    if a.dry_run:
        _log("dry run: nothing started")
        return 0
    pf = dk.preflight(a.image, a.results_dir, allow_arm=a.allow_arm)
    _log(f"docker: {pf.info}")
    for w in pf.warnings:
        _log(f"warning: {w}")
    if not pf.ok:
        for p in pf.problems:
            _log(f"problem: {p}")
        return 2
    (Path(a.results_dir) / "logs").mkdir(parents=True, exist_ok=True)
    rc = dk.execute(cmd, log=_log)
    out = dk.final_results_path(a.results_dir, a.run_id)
    _log(f"harness exit code {rc}; results: {out}" + ("" if out.exists() else " (missing!)"))
    if out.exists():
        _log(f"next: keel eval abs-labels <records> --results-dir {a.results_dir} --run-id {a.run_id}")
    return rc


def cmd_abs_docker_build(a: argparse.Namespace) -> int:
    from . import abs_docker as dk
    cmd = dk.build_command(a.tag, a.sweabs_ref, a.lfs, no_cache=a.no_cache)
    print(" ".join(cmd))
    if a.dry_run:
        return 0
    pf = dk.preflight(a.tag, allow_arm=a.allow_arm)
    blocking = [p for p in pf.problems if "not found; build it" not in p]  # a missing image is why we build
    if blocking:
        for p in blocking:
            _log(f"problem: {p}")
        return 2
    return dk.execute(cmd, log=_log)


def cmd_docker_check(a: argparse.Namespace) -> int:
    from . import abs_docker as dk
    pf = dk.preflight(a.image, a.results_dir, allow_arm=a.allow_arm)
    _log(f"docker: {pf.info}")
    for w in pf.warnings:
        _log(f"warning: {w}")
    for p in pf.problems:
        _log(f"problem: {p}")
    _log("ready for Oracle 2" if pf.ok else "not ready for Oracle 2")
    return 0 if pf.ok else 2


def cmd_abs_labels(a: argparse.Namespace) -> int:
    from . import abs_labels as ab
    recs = rec_io.load(a.records)
    if a.results_dir and not a.final_results:
        from .abs_docker import final_results_path
        if not a.run_id:
            raise SystemExit("--results-dir needs --run-id")
        a.final_results = str(final_results_path(a.results_dir, a.run_id))
    if a.sweabs_dir and not a.final_results:
        if not a.run_id:
            raise SystemExit("--sweabs-dir needs --run-id")
        a.final_results = str(ab.final_results_path(a.sweabs_dir, a.run_id))
    if a.final_results:
        labels = ab.read_final_results(a.final_results)
        for model, prob in ab.final_results_problems(a.final_results).items():
            if prob["error"]:
                _log(f"{model}: SWE-ABS reported errors on {len(prob['error'])} instances (no ABS label for them)")
    else:
        labels = ab.read_harness_reports(a.workdir, a.run_id)
    key_map = json.loads(Path(a.key_map).read_text()) if a.key_map else None
    ab.merge(recs, labels, key_map=key_map, monotone=not a.no_monotone, log=_log)
    rec_io.save(a.out or a.records, recs)
    _log(f"wrote {a.out or a.records}")
    return 0


def cmd_audit_export(a: argparse.Namespace) -> int:
    from .audit import export
    recs = _load_all(a.records)
    st = export(recs, _instances(a), a.out, a.audit_frac, tuple(a.oracles.split(",")), a.double_frac, a.seed)
    _log(f"audit: {st['items']} items ({st['double_rated']} for a second rater) in {a.out}. Give rater 1 "
         f"items/ + rater1.csv and rater 2 items/ + rater2.csv, separately; keep key.csv away from both.")
    return 0


def cmd_audit_status(a: argparse.Namespace) -> int:
    from .audit import audit_status
    st = audit_status(a.audit)
    _log(f"audit status: {st}")
    if st["disagreements"] and st["adjudicated"] < st["disagreements"]:
        _log(f"{st['disagreements'] - st['adjudicated']} disagreements need a `final` verdict in "
             f"{Path(a.audit) / 'adjudication.csv'}")
    if st["kappa"] is not None and st["kappa"] < 0.6:
        _log("Cohen's kappa is below 0.6: refine the labelling guide and re-rate")
    return 0


def cmd_labels_status(a: argparse.Namespace) -> int:
    from .thesis import readiness
    for p in a.records:
        st = readiness(rec_io.load(p), a.audit_frac)
        _log(f"{p}: {st}")
    return 0


def cmd_audit_import(a: argparse.Namespace) -> int:
    from .audit import import_verdicts
    recs = rec_io.load(a.records)
    st = import_verdicts(recs, a.audit, unsure_as_wrong=not a.unsure_as_missing)
    rec_io.save(a.out or a.records, recs)
    _log(f"audit import: {st}")
    if st.get("kappa") is not None and st["kappa"] < 0.6:
        _log("Cohen's kappa is below 0.6: refine the labelling guide and re-rate before using these labels")
    return 0


def cmd_thesis(a: argparse.Namespace) -> int:
    """RQ1 (inflation), RQ3 (correction) and RQ4 (audit budget) in one run, with tables and charts."""
    from .figures import write_thesis_outputs
    from .records import has_oracle
    from .thesis import audit_budget_curve, evaluate_noisy, inflation
    recs = _load_all(a.records)
    feats = _features(a.features, recs)
    alphas = [float(x) for x in a.alphas.split(",")]
    eval_label = a.eval_label
    if eval_label == "auto":
        eval_label = "true" if has_oracle(recs, "true") else "best"
    _log(f"{len(recs)} bugs; evaluating against oracle '{eval_label}'")
    res = {"n_bugs": len(recs), "eval_label": eval_label, "n_splits": a.splits, "alphas": alphas,
           "simulated": any(r.get("simulated") for r in recs)}
    oracles = tuple(o for o in ("orig", "abs", "human", "true")
                    if any(rec_label_any(r, o) for r in recs))
    res["inflation"] = inflation(recs, alphas, a.splits, a.delta, a.seed, feats, oracles)
    res["correction"] = evaluate_noisy(recs, alphas, a.splits, a.delta, a.seed, feats, eval_label, a.audit_frac)
    if a.budgets:
        res["budget"] = audit_budget_curve(recs, [int(x) for x in a.budgets.split(",")], a.budget_alpha,
                                           max(20, a.splits // 2), a.delta, a.seed, feats, eval_label, a.audit_frac)
    for p in write_thesis_outputs(res, a.out_dir):
        _log(f"wrote {p}")
    return 0


def rec_label_any(record: dict, oracle: str) -> bool:
    from .records import label as rec_label
    return any(rec_label(c, oracle) is not None for c in record["candidates"])


def _gen_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models", default="muse-spark,muse-spark-open")
    p.add_argument("--attacker", default="muse-spark")
    p.add_argument("--no-attack", action="store_true")
    p.add_argument("--triage")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--cache", default=".keel-cache", help="model response cache")


def add_eval_commands(sub) -> None:
    ev = sub.add_parser("eval", help="benchmarks and analysis")
    es = ev.add_subparsers(dest="eval_cmd", required=True)

    s = es.add_parser("simulate", help="write SIMULATED records (for tests and the offline demo only)")
    s.add_argument("--out", default="results/simulated/records.jsonl")
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--eps-fc", type=float, default=0.0, help="P(held-out tests pass a wrong patch)")
    s.add_argument("--eps-fw", type=float, default=0.0, help="P(held-out tests fail a correct patch)")
    s.add_argument("--abs-catch", type=float, default=0.7,
                   help="share of slipped-through wrong patches that strengthened tests catch")
    s.add_argument("--human-error", type=float, default=0.0, help="P(a human auditor flips the label)")
    s.set_defaults(fn=cmd_simulate)

    q = es.add_parser("quixbugs", help="run Keel on QuixBugs and record hidden-test labels")
    _gen_args(q)
    q.add_argument("--work", default="results/quixbugs/work")
    q.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "keel"))
    q.add_argument("--out", default="results/quixbugs/records.jsonl")
    q.add_argument("--reports", default="results/quixbugs/reports")
    q.add_argument("--only", help="comma-separated program names")
    q.add_argument("--rebuild", action="store_true")
    q.set_defaults(fn=cmd_quixbugs)

    w = es.add_parser("swebench", help="SWE-bench Lite/Verified with Keel's generators: generate / harness / records")
    _gen_args(w)
    w.add_argument("--dataset", choices=["lite", "verified"], default="lite")
    w.add_argument("--stage", choices=["generate", "harness", "records", "all", "list-runs"], default="all")
    w.add_argument("--run-ids", help="harness stage: only these run ids (comma-separated)")
    w.add_argument("--low-disk", action="store_true", help="harness stage: delete images after each instance")
    w.add_argument("--n", type=int, default=100, help="instances (fixed-seed random subset)")
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "keel"))
    w.add_argument("--candidates", default="results/swebench/candidates.jsonl")
    w.add_argument("--workdir", default="results/swebench/harness")
    w.add_argument("--workers", type=int, default=4)
    w.add_argument("--out", default="results/swebench/records.jsonl")
    w.set_defaults(fn=cmd_swebench)

    an = es.add_parser("analyze", help="repeated-split analysis -> dashboard JSON")
    an.add_argument("records", nargs="+")
    an.add_argument("--out", default="web/public/data/results.json")
    an.add_argument("--splits", type=int, default=200)
    an.add_argument("--delta", type=float, default=0.1)
    an.add_argument("--seed", type=int, default=0)
    an.add_argument("--features", choices=["auto", "keel", "external"], default="auto")
    an.add_argument("--cal-label", default="orig", help="oracle to fit and calibrate on (orig, abs, human, best)")
    an.add_argument("--eval-label", default="orig", help="oracle to measure realised error with")
    an.set_defaults(fn=cmd_analyze)

    sm = es.add_parser("submission", help="records from published SWE-bench submissions (swe-bench/experiments)")
    sm.add_argument("folders", nargs="+", help="submission folders, e.g. evaluation/verified/<name>")
    sm.add_argument("--instances", help="local SWE-bench rows (.json/.jsonl); default: download")
    sm.add_argument("--dataset", choices=["lite", "verified"], default="verified")
    sm.add_argument("--out-dir", default="results/agents")
    sm.add_argument("--limit", type=int, help="only the first N instances (pilot)")
    sm.add_argument("--judge", help="model for the LLM-judge signal (optional, costs API calls)")
    sm.add_argument("--cache", default=".keel-cache")
    sm.add_argument("--no-traj", action="store_true")
    sm.add_argument("--preds", help="predictions file if it is not in the folder (all_preds.jsonl or preds.json)")
    sm.set_defaults(fn=cmd_submission)

    ins = es.add_parser("instances", help="save SWE-bench rows locally (needed by fetch --limit, submission, audit)")
    ins.add_argument("--dataset", choices=["lite", "verified"], default="verified")
    ins.add_argument("--out", default="results/verified.jsonl")
    ins.set_defaults(fn=cmd_instances)

    fe = es.add_parser("fetch", help="Oracle 1: download published submissions (patches, reports, results.json)")
    fe.add_argument("submissions", nargs="+", help="submission names, e.g. 20250807_openhands_gpt5")
    fe.add_argument("--split", default="verified")
    fe.add_argument("--out", default="data/submissions", help="also the SWE-ABS --vaild_model_path")
    fe.add_argument("--instances", help="only instances in this rows file")
    fe.add_argument("--limit", type=int, help="only the first N instances (pilot; needs --instances)")
    fe.add_argument("--trajs", action="store_true", help="also download trajectories (for trajectory signals)")
    fe.add_argument("--workers", type=int, default=8)
    fe.set_defaults(fn=cmd_fetch)

    ar = es.add_parser("abs-run", help="Oracle 2: build / run the SWE-ABS harness on downloaded submissions")
    ar.add_argument("--models-root", default="data/submissions", help="the `keel eval fetch` output folder")
    ar.add_argument("--models", required=True, help="comma-separated submission names")
    ar.add_argument("--run-id", required=True)
    ar.add_argument("--sweabs-dir", help="SWE-ABS checkout; without it the command is only printed")
    ar.add_argument("--pilot", type=int, help="only the first N instances the first model resolved")
    ar.add_argument("--instance-ids", help="comma-separated instance ids")
    ar.add_argument("--workers", type=int, default=8)
    ar.add_argument("--timeout", type=int, default=1800, help="per-instance timeout in seconds (SWE-ABS default)")
    ar.add_argument("--python", default="python", help="python of the swe-abs environment")
    ar.add_argument("--rerun", action="store_true", help="ignore an existing final_results.json for this run id")
    ar.add_argument("--dry-run", action="store_true")
    ar.add_argument("--docker", action="store_true",
                    help="run the harness in the keel-oracle2 image (no SWE-ABS install needed)")
    ar.add_argument("--image", default="keel-oracle2", help="with --docker: image tag")
    ar.add_argument("--results-dir", default="results/sweabs", help="with --docker: where final_results.json goes")
    ar.add_argument("--allow-arm", action="store_true", help="with --docker: try on an ARM engine anyway")
    ar.set_defaults(fn=cmd_abs_run)

    adb = es.add_parser("abs-docker-build", help="Oracle 2: build the keel-oracle2 image (SWE-ABS + Keel)")
    adb.add_argument("--tag", default="keel-oracle2")
    adb.add_argument("--sweabs-ref", help="SWE-ABS commit to install (default: the pinned one in the Dockerfile)")
    adb.add_argument("--lfs", action="store_true", help="also fetch SWE-ABS's Git LFS files")
    adb.add_argument("--no-cache", action="store_true")
    adb.add_argument("--allow-arm", action="store_true")
    adb.add_argument("--dry-run", action="store_true")
    adb.set_defaults(fn=cmd_abs_docker_build)

    dc = es.add_parser("docker-check", help="Oracle 2: is Docker ready (Linux engine, x86-64, resources, image)?")
    dc.add_argument("--image", default="keel-oracle2")
    dc.add_argument("--results-dir", default="results/sweabs")
    dc.add_argument("--allow-arm", action="store_true")
    dc.set_defaults(fn=cmd_docker_check)

    ab = es.add_parser("abs-labels", help="merge strengthened-test (SWE-ABS) labels into records")
    ab.add_argument("records")
    ab.add_argument("--sweabs-dir", help="find final_results.json for --run-id in this SWE-ABS checkout")
    ab.add_argument("--results-dir", help="find final_results.json for --run-id here (abs-run --docker output)")
    ab.add_argument("--final-results", help="SWE-ABS swe_abs_res/eval_agent/<run>/final_results.json")
    ab.add_argument("--workdir", help="official-harness workdir (cross-check route)")
    ab.add_argument("--run-id")
    ab.add_argument("--key-map", help="JSON {generator: SWE-ABS model key}")
    ab.add_argument("--no-monotone", action="store_true", help="allow ABS to pass a patch the original tests fail")
    ab.add_argument("--out", help="default: overwrite the records file")
    ab.set_defaults(fn=cmd_abs_labels)

    ae = es.add_parser("audit-export", help="write the human-audit items for the audit pool")
    ae.add_argument("records", nargs="+")
    ae.add_argument("--instances", help="local SWE-bench rows (.json/.jsonl); default: download")
    ae.add_argument("--dataset", choices=["lite", "verified"], default="verified")
    ae.add_argument("--out", default="results/audit")
    ae.add_argument("--audit-frac", type=float, default=0.25)
    ae.add_argument("--oracles", default="orig,abs")
    ae.add_argument("--double-frac", type=float, default=0.3)
    ae.add_argument("--seed", type=int, default=0)
    ae.set_defaults(fn=cmd_audit_export)

    ai = es.add_parser("audit-import", help="read human verdicts into records (and report Cohen's kappa)")
    ai.add_argument("records")
    ai.add_argument("--audit", default="results/audit")
    ai.add_argument("--out", help="default: overwrite the records file")
    ai.add_argument("--unsure-as-missing", action="store_true", help="default treats 'unsure' as wrong")
    ai.set_defaults(fn=cmd_audit_import)

    ast_ = es.add_parser("audit-status", help="audit progress and kappa; writes adjudication.csv for disagreements")
    ast_.add_argument("--audit", default="results/audit")
    ast_.set_defaults(fn=cmd_audit_status)

    ls = es.add_parser("labels-status", help="labels per oracle and which thesis variants can run")
    ls.add_argument("records", nargs="+")
    ls.add_argument("--audit-frac", type=float, default=0.25)
    ls.set_defaults(fn=cmd_labels_status)

    th = es.add_parser("thesis", help="RQ1 inflation, RQ3 correction, RQ4 audit budget -> tables and charts")
    th.add_argument("records", nargs="+")
    th.add_argument("--out-dir", default="results/thesis")
    th.add_argument("--alphas", default="0.05,0.1,0.15,0.2,0.3")
    th.add_argument("--splits", type=int, default=200)
    th.add_argument("--delta", type=float, default=0.1)
    th.add_argument("--seed", type=int, default=0)
    th.add_argument("--features", choices=["auto", "keel", "external"], default="auto")
    th.add_argument("--eval-label", default="auto", help="auto = 'true' in simulation, else 'best'")
    th.add_argument("--audit-frac", type=float, default=0.25)
    th.add_argument("--budgets", default="30,60,100,150,300", help="audit budgets for RQ4 ('' to skip)")
    th.add_argument("--budget-alpha", type=float, default=0.1)
    th.set_defaults(fn=cmd_thesis)

    c = sub.add_parser("calibrate", help="build a deployment calibration from records")
    c.add_argument("records", nargs="+")
    c.add_argument("--out", default="calibration.json")
    c.add_argument("--fit-fraction", type=float, default=0.4)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--models", default="muse-spark,muse-spark-open")
    c.add_argument("--k", type=int, default=5)
    c.add_argument("--no-attack", action="store_true")
    c.add_argument("--triage")
    c.add_argument("--fingerprint", help="override the computed fingerprint")
    c.add_argument("--features", choices=["auto", "keel", "external"], default="auto")
    c.set_defaults(fn=cmd_calibrate)
