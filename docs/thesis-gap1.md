# Thesis workflow: certified patch acceptance under imperfect test oracles

This is the code side of the thesis plan (Gap 1). It turns published agent patches into records with
three labels per patch, runs the experiments for RQ1, RQ3 and RQ4, and writes every table and chart.

| Piece | Where |
|---|---|
| Fixed-sequence Learn-then-Test, false-merge target, three tiers | `keel/conformal.py` |
| Noise-corrected LTT (variants A and B) | `keel/conformal_noisy.py` |
| Labels per oracle (`orig`, `abs`, `human`, `true`, `best`) | `keel/eval/records.py` |
| Records from leaderboard submissions | `keel/eval/experiments_repo.py` |
| Signals for external agents (diff, issue, trajectory, LLM judge) | `keel/eval/signals_ext.py` |
| Strengthened-test labels (SWE-ABS) | `keel/eval/abs_labels.py` |
| Human audit: export, import, Cohen's kappa | `keel/eval/audit.py` |
| Experiments: inflation, correction, audit budget | `keel/eval/thesis.py` |
| Tables and SVG charts | `keel/eval/figures.py` |
| Simulation with oracle noise (known ground truth) | `keel/eval/simulate.py` |
| Tests | `tests/test_thesis.py` |

## The guarantee in one paragraph

A wrong patch either fails the oracle (the held-out tests) or slips past it, so
`FMR_true(λ) ≤ P(oracle says wrong | s ≤ λ) + P(wrong but passes | s ≤ λ)`. The first term is bounded
from the calibration set, the second from an independent human audit, each with an exact
(Clopper–Pearson) bound. Fixed-sequence testing over a grid chosen on the fit split then gives
`P(FMR_true(λ̂) ≤ α) ≥ 1 − δ₁ − δ₂`, assuming exchangeability, an independent audit sample and correct
human labels. Variant A uses the original tests as the oracle; variant B uses the strengthened tests,
so the audit only has to catch what even those miss.

## Step 0: try it on simulated data (no downloads)

```bash
keel eval simulate --n 1500 --eps-fc 0.3 --out results/sim/noisy.jsonl
keel eval thesis results/sim/noisy.jsonl --out-dir results/thesis-sim
```

`results/thesis-sim/README.md` then shows RQ1 (plain LTT breaks against the true labels), RQ3 (variants
A and B stay within δ) and RQ4 (merge rate against audit budget). Simulated records are always marked.

## Step 1: published agent patches and original labels (no Docker)

```bash
git clone --depth 1 https://github.com/swe-bench/experiments
keel eval submission experiments/evaluation/verified/<submission> [more folders ...] \
    --dataset verified --out-dir results/agents            # --limit 100 for the pilot
```

One records file per submission. `orig` = not resolved according to the submission's
`results/results.json` (or its per-instance `logs/*/report.json`). Check first that the folders you
pick have these files; some trajectories are only in a separate download. Add `--judge muse-spark`
(or any configured model) for the LLM-judge signal; it costs one call per patch.

## Step 2: strengthened-test labels (Docker, Linux x86)

Run the SWE-ABS harness (github.com/OpenAgentEval/SWE-ABS) on the same submission folders; its
README shows the command (`python -m swebench.runtest.run_evaluation_test ... --vaild_model_name
<submission>`). Then merge its output:

```bash
keel eval abs-labels results/agents/<submission>.jsonl \
    --final-results swe-bench/swe_abs_res/eval_agent/<run_id>/final_results.json
```

If the generator name differs from the SWE-ABS model key, pass `--key-map map.json`
(`{"<generator>": "<model key>"}`). A patch that fails the original tests stays wrong (strengthened
tests only add tests); the command reports how often the two runs disagree that way, which is an
environment check worth reporting.

## Step 3: human audit

```bash
keel eval audit-export results/agents/*.jsonl --dataset verified --out results/audit
```

This writes one Markdown file per item (issue, patch, reference patch, checklist), a blind
`sheet.csv` for the raters and a private `key.csv`. Only audit-pool patches (a fixed 25% of issues,
chosen by hash) that pass an oracle are listed; patches the tests already reject count as wrong.
About 30% of items are marked for a second rater. Fill `rater_1`, `rater_2` and, for disagreements,
`final` with `correct`, `wrong` or `unsure`, then:

```bash
keel eval audit-import results/agents/<submission>.jsonl --audit results/audit
```

It prints Cohen's kappa and refuses nothing, but warns below 0.6. `unsure` counts as wrong unless
`--unsure-as-missing`.

## Step 4: experiments

```bash
keel eval thesis results/agents/<submission>.jsonl --out-dir results/thesis/<submission> \
    --alphas 0.05,0.1,0.15,0.2,0.3 --budgets 30,60,100,150
```

* RQ1 `rq1_inflation.*`: plain LTT calibrated on the original tests, judged by each oracle.
* RQ3 `rq3_*`: plain, plain on strengthened tests, variant A, variant B and the baselines; realised
  false-merge rate, violation frequency P(FMR > α) and automation, judged by `best` (human where
  audited, else strengthened tests, else original).
* RQ4 `rq4_budget.*`: automation of A and B as the audit budget grows.

`keel eval analyze --cal-label orig --eval-label abs` gives the dashboard-style analysis with any pair
of oracles. RQ2 (which signals separate silent failures) uses the same records; the ablation table in
`keel eval analyze` and `--features external` are the starting point.

## What is not automated

* Running the SWE-ABS harness (Docker, outside this repository).
* The human verdicts.
* Variant C (prediction-powered inference, Einbinder et al., TPAMI 2025) is not implemented.
