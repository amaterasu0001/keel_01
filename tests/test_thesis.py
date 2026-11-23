"""Tests for the thesis additions: fixed-sequence LTT, tiers, noise-corrected LTT, labels, adapters."""

import json
import random

import pytest

from keel import conformal, conformal_noisy
from keel.calibration import Calibration
from keel.eval import abs_labels, audit, experiments_repo
from keel.eval import signals_ext as sx
from keel.eval.records import label, set_label
from keel.eval.simulate import simulate
from keel.eval.thesis import audit_requests, evaluate_noisy, in_audit_pool
from keel.pipeline import Pipeline
from keel.scoring import Scorer
from keel.types import AUTO_MERGE, NEEDS_REVIEW, Candidate, CandidateEvaluation


def _draw(rng, n, base_wrong=0.4):
    out = []
    for _ in range(n):
        wrong = rng.random() < base_wrong
        s = min(0.999, max(0.0, rng.gauss(0.6 if wrong else 0.25, 0.2)))
        out.append((s, wrong))
    return out


# --- conformal ---------------------------------------------------------------------------------

def test_binom_ucb_matches_pvalue():
    for w, k in [(0, 10), (1, 20), (3, 100), (0, 23)]:
        u = conformal.binom_ucb(w, k, 0.1)
        assert conformal.binom_pvalue(w, k, u + 1e-6) <= 0.1 + 1e-9
        assert conformal.binom_pvalue(w, k, u - 1e-3) > 0.1
    assert conformal.binom_ucb(0, 0, 0.1) == 1.0


def test_fixed_sequence_certifies_what_bonferroni_cannot():
    # The QuixBugs calibration: 23 identical scores, all correct, one gated wrong patch.
    scores, wrong = [0.3775] * 23 + [1.0], [False] * 23 + [True]
    grid = [0.3775]
    assert conformal.ltt_threshold(scores, wrong, 0.1, 0.1, grid) == 0.3775            # 0.9^23 = 0.089 <= 0.1
    assert conformal.ltt_threshold(scores, wrong, 0.1, 0.1) == conformal.NEVER         # default grid starts at 0.005
    assert conformal.ltt_threshold(scores, wrong, 0.1, 0.1, procedure="bonferroni") == conformal.NEVER


def test_fixed_sequence_ltt_with_fit_grid_controls_fdr():
    rng = random.Random(7)
    alpha, delta, trials, viol = 0.15, 0.1, 300, 0
    for _ in range(trials):
        fit, cal, test = _draw(rng, 200), _draw(rng, 300), _draw(rng, 2000)
        grid = conformal.grid_from_scores([s for s, _ in fit])
        grid = conformal.order_grid(grid, [s for s, _ in fit], [w for _, w in fit], len(cal), delta)
        thr = conformal.ltt_threshold([s for s, _ in cal], [w for _, w in cal], alpha, delta, grid)
        st = conformal.rule_stats([s for s, _ in test], [w for _, w in test], thr)
        viol += st.n_merged > 0 and st.fdr > alpha
    assert viol / trials <= delta


def test_order_grid_skips_starved_start():
    rng = random.Random(3)
    fit = _draw(rng, 400)
    grid = conformal.grid_from_scores([s for s, _ in fit])
    ordered = conformal.order_grid(grid, [s for s, _ in fit], [w for _, w in fit], 300, 0.1)
    assert ordered and ordered[0] >= grid[0] and ordered == sorted(ordered)


def test_tiers_jointly_valid_and_ordered():
    rng = random.Random(11)
    alpha, beta, delta, trials, bad = 0.15, 0.15, 0.1, 200, 0
    for _ in range(trials):
        cal, test = _draw(rng, 400), _draw(rng, 3000)
        t = conformal.tier_thresholds([s for s, _ in cal], [w for _, w in cal], alpha, beta, delta)
        assert t.merge < t.reject
        merged = [w for s, w in test if t.tier(s) == conformal.MERGE]
        rejected = [w for s, w in test if t.tier(s) == conformal.REJECT]
        fmr = sum(merged) / len(merged) if merged else 0
        fcr = sum(1 for w in rejected if not w) / len(rejected) if rejected else 0
        bad += fmr > alpha or fcr > beta
    assert bad / trials <= delta


def test_tier_gated_is_rejected_only_when_certified():
    assert conformal.Tiers(merge=0.1, reject=0.9).tier(0.5, gated=True) == conformal.REJECT
    assert conformal.Tiers(merge=0.1, reject=conformal.NO_REJECT).tier(0.5, gated=True) == conformal.REVIEW
    assert conformal.Tiers(merge=0.1, reject=0.9).tier(0.05) == conformal.MERGE


# --- noise-corrected LTT -----------------------------------------------------------------------

def _noisy_draw(rng, n, eps_fc=0.4):
    """(score, oracle_wrong, true_wrong): wrong patches slip past the oracle with prob eps_fc."""
    out = []
    for _ in range(n):
        true = rng.random() < 0.35
        s = min(0.999, max(0.0, rng.gauss(0.55 if true else 0.25, 0.18)))
        oracle = true and rng.random() >= eps_fc
        out.append((s, oracle, true))
    return out


def test_noise_corrected_ltt_controls_true_fmr_where_plain_fails():
    rng = random.Random(5)
    alpha, d1, d2, trials = 0.15, 0.05, 0.05, 150
    viol_plain = viol_corr = merged_corr = 0
    for _ in range(trials):
        fit, cal, aud, test = (_noisy_draw(rng, n) for n in (300, 400, 300, 3000))
        grid = conformal.grid_from_scores([s for s, *_ in fit])
        g1 = conformal.order_grid(grid, [s for s, *_ in fit], [o for _, o, _ in fit], len(cal), 0.1)
        plain = conformal.ltt_threshold([s for s, *_ in cal], [o for _, o, _ in cal], alpha, 0.1, g1)
        g2 = conformal.order_grid(grid, [s for s, *_ in fit], [o for _, o, _ in fit], len(cal), d1, len(aud), d2)
        missed = conformal_noisy.missed_flags([o for _, o, _ in aud], [t for *_, t in aud])
        corr = conformal_noisy.noise_corrected_threshold([s for s, *_ in cal], [o for _, o, _ in cal],
                                                         [s for s, *_ in aud], missed, alpha, d1, d2, g2)
        for thr, is_corr in ((plain, False), (corr, True)):
            st = conformal.rule_stats([s for s, *_ in test], [t for *_, t in test], thr)
            v = st.n_merged > 0 and st.fdr > alpha
            if is_corr:
                viol_corr += v
                merged_corr += st.n_merged > 0
            else:
                viol_plain += v
    assert viol_corr / trials <= d1 + d2
    assert viol_plain / trials > 0.5          # the inflation the thesis measures
    assert merged_corr / trials > 0.5         # and the corrected gate still merges something


def test_missed_flags_needs_human_label_only_for_oracle_passes():
    assert conformal_noisy.missed_flags([True, False], [None, True]) == [False, True]
    with pytest.raises(ValueError):
        conformal_noisy.missed_flags([False], [None])


# --- records / simulation ----------------------------------------------------------------------

def test_labels_and_best_oracle():
    c = {"wrong": False}
    assert label(c, "orig") is False and label(c, "abs") is None and label(c, "best") is False
    set_label(c, "abs", True)
    assert label(c, "best") is True
    set_label(c, "human", False)
    assert label(c, "best") is False and c["labels"]["orig"] is False


def test_simulate_noise_keeps_features_and_defaults():
    a, b = simulate(40, seed=2), simulate(40, seed=2, eps_fc=0.4)
    ca = [c for r in a for c in r["candidates"]]
    cb = [c for r in b for c in r["candidates"]]
    assert [c["features"] for c in ca] == [c["features"] for c in cb]
    assert all(c["wrong"] == c["labels"]["true"] for c in ca)
    assert sum(c["wrong"] for c in cb) < sum(c["labels"]["true"] for c in cb)
    assert all(c["labels"]["abs"] >= c["labels"]["orig"] for c in cb)  # stronger oracle only adds failures


def test_evaluate_noisy_on_simulation():
    recs = simulate(500, seed=9, eps_fc=0.35)
    ev = evaluate_noisy(recs, alphas=[0.2], n_splits=15, eval_label="true", audit_frac=0.25)
    assert {"plain", "A", "B"} <= set(ev["methods"])
    assert ev["methods"]["A"][0]["viol_fdr"] <= ev["methods"]["plain"][0]["viol_fdr"]


def test_audit_pool_is_deterministic():
    ids = [f"t{i}" for i in range(400)]
    pool = [i for i in ids if in_audit_pool(i, 0.25)]
    assert pool == [i for i in ids if in_audit_pool(i, 0.25)]
    assert 60 < len(pool) < 140


# --- adapters ----------------------------------------------------------------------------------

DIFF = """diff --git a/pkg/core.py b/pkg/core.py
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1,3 +1,3 @@ def parse_header(value):
-    return value.split(",")
+    return [v.strip() for v in value.split(",")]
"""


def _submission(tmp_path):
    sub = tmp_path / "20250101_agent"
    (sub / "results").mkdir(parents=True)
    (sub / "trajs").mkdir()
    (sub / "all_preds.jsonl").write_text(
        json.dumps({"instance_id": "o__r-1", "model_patch": DIFF}) + "\n"
        + json.dumps({"instance_id": "o__r-2", "model_patch": DIFF}) + "\n")
    (sub / "results" / "results.json").write_text(json.dumps({"resolved": ["o__r-1"], "no_apply": []}))
    (sub / "trajs" / "o__r-1.traj").write_text(json.dumps({"trajectory": [
        {"action": "open pkg/core.py"}, {"action": "edit 1:3"}, {"action": "python -m pytest tests"}]}))
    insts = {i: {"instance_id": i, "repo": "o/r", "problem_statement": "parse_header keeps spaces in core.py",
                 "patch": DIFF} for i in ("o__r-1", "o__r-2", "o__r-3")}
    return sub, insts


def test_load_submission_labels_and_features(tmp_path):
    sub, insts = _submission(tmp_path)
    recs = experiments_repo.load_submission(sub, insts, log=lambda m: None)
    by = {r["task_id"]: r["candidates"][0] for r in recs}
    assert by["o__r-1"]["wrong"] is False and by["o__r-2"]["wrong"] is True
    assert by["o__r-3"]["gates"] == ["no_patch"]
    f = by["o__r-1"]["features"]
    assert f["has_traj"] == 1.0 and f["traj_test_runs"] == 1.0 and f["issue_file_mention"] == 1.0
    assert set(sx.EXTERNAL_FEATURES) <= set(f)


def test_abs_merge_is_monotone(tmp_path):
    sub, insts = _submission(tmp_path)
    recs = experiments_repo.load_submission(sub, insts, log=lambda m: None)
    fr = tmp_path / "final_results.json"
    fr.write_text(json.dumps({"o__r-1": {"pass": [], "fail": ["20250101_agent"]},
                              "o__r-2": {"pass": ["20250101_agent"], "fail": []}}))
    st = abs_labels.merge(recs, abs_labels.read_final_results(fr), log=lambda m: None)
    by = {r["task_id"]: r["candidates"][0] for r in recs}
    assert by["o__r-1"]["labels"]["abs"] is True       # silent failure caught by stronger tests
    assert by["o__r-2"]["labels"]["abs"] is True       # orig fail stays fail (monotone)
    assert st["flipped_to_wrong"] == 1 and st["abs_pass_but_orig_fail"] == 1


def test_audit_roundtrip_and_kappa(tmp_path):
    recs = simulate(200, seed=1)
    insts = {r["task_id"]: {"repo": "sim", "problem_statement": "x", "patch": ""} for r in recs}
    st = audit.export(recs, insts, tmp_path / "aud", audit_frac=0.3, oracles=("orig",))
    reqs = audit_requests(recs, 0.3, ("orig",))
    assert st["items"] == len(reqs) > 0
    rows = list(__import__("csv").DictReader(open(tmp_path / "aud" / "sheet.csv", encoding="utf-8")))
    for i, r in enumerate(rows):
        r["rater_1"] = "correct" if i % 3 else "wrong"
        if r["rater_2"] != "-":
            r["rater_2"] = r["rater_1"]
    with open(tmp_path / "aud" / "sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = __import__("csv").DictWriter(f, fieldnames=audit.SHEET_FIELDS)
        w.writeheader()
        w.writerows(rows)
    res = audit.import_verdicts(recs, tmp_path / "aud")
    assert res["labelled"] == len(rows) and res["kappa"] == 1.0
    assert audit.cohen_kappa(["a", "b", "a", "b"], ["a", "a", "b", "b"]) == 0.0


def test_issue_and_trajectory_features():
    f = sx.issue_features("The parse_header function in core.py keeps spaces", DIFF)
    assert f["issue_token_overlap"] > 0 and f["issue_file_mention"] == 1.0
    assert sx.trajectory_features(None)["has_traj"] == 0.0
    t = sx.trajectory_features([{"role": "assistant", "content": "pytest -q"}])
    assert t["traj_test_runs"] == 1.0


# --- pipeline decision -------------------------------------------------------------------------

def _ev(score, gates=()):
    ev = CandidateEvaluation(candidate=Candidate("c0", "m", 0, DIFF), applied=True)
    ev.score, ev.gate_failures = score, list(gates)
    return ev


def test_pipeline_three_tiers():
    # 60 good patches score 0.01, 40 bad ones 0.95, 30 gated (score 1.0, wrong) as in real calibrations.
    cal = Calibration(scorer=Scorer(), scores=[0.01] * 60 + [0.95] * 40 + [1.0] * 30,
                      wrong=[False] * 60 + [True] * 70, fingerprint="", grid=[0.01, 0.5, 0.95])
    pipe = Pipeline([], calibration=cal, alpha=0.1, method="ltt", delta=0.1, beta=0.1, log=lambda m: None)
    d = pipe.decide(_ev(0.01))
    assert d.label == AUTO_MERGE and d.tier == "merge"
    d = pipe.decide(_ev(0.7))
    assert d.label == NEEDS_REVIEW and d.tier == "review"
    d = pipe.decide(_ev(0.97))
    assert d.label == NEEDS_REVIEW and d.tier == "reject" and d.reject_threshold is not None
    assert d.reasons[0].startswith("Likely wrong")


def test_thesis_cli_end_to_end(tmp_path):
    import argparse

    from keel.eval import records as rec_io
    from keel.eval.cli import add_eval_commands
    rec_io.save(tmp_path / "sim.jsonl", simulate(300, seed=4, eps_fc=0.3))
    p = argparse.ArgumentParser()
    add_eval_commands(p.add_subparsers(dest="cmd"))
    a = p.parse_args(["eval", "thesis", str(tmp_path / "sim.jsonl"), "--out-dir", str(tmp_path / "out"),
                      "--splits", "4", "--alphas", "0.1,0.2", "--budgets", "20,40"])
    assert a.fn(a) == 0
    res = json.loads((tmp_path / "out" / "results.json").read_text())
    assert res["eval_label"] == "true" and "A" in res["correction"]["methods"] and res["budget"]
    assert (tmp_path / "out" / "rq1_inflation.svg").exists() and (tmp_path / "out" / "README.md").exists()
