# Thesis workflow: certified patch acceptance under imperfect test oracles

This is the code side of the thesis plan (Gap 1). It turns published agent patches into records with
three labels per patch, runs the experiments for RQ1, RQ3 and RQ4, and writes every table and chart.

| Piece | Where |
|---|---|
| Fixed-sequence Learn-then-Test, false-merge target, three tiers | `keel/conformal.py` |
| Noise-corrected LTT (variants A and B) | `keel/conformal_noisy.py` |
| Labels per oracle (`orig`, `abs`, `human`, `true`, `best`) | `keel/eval/records.py` |
| Download submissions and SWE-bench rows (Oracle 1 input, SWE-ABS layout) | `keel/eval/fetch.py` |
| Records from leaderboard submissions | `keel/eval/experiments_repo.py` |
| Signals for external agents (diff, issue, trajectory, LLM judge) | `keel/eval/signals_ext.py` |
| Strengthened-test labels: run SWE-ABS, read its output (Oracle 2) | `keel/eval/abs_labels.py` |
| Oracle 2 in Docker: image, preflight checks, `docker run` | `docker/oracle2/Dockerfile`, `keel/eval/abs_docker.py` |
| Human audit: two blind sheets, status, adjudication, import, kappa (Oracle 3) | `keel/eval/audit.py` |
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

## Step 1 · Oracle 1: published agent patches and original labels (no Docker)

```bash
keel eval instances --out results/verified.jsonl                    # the 500 SWE-bench Verified rows, once
keel eval fetch 20250807_openhands_gpt5 20241202_agentless-1.5_claude-3.5-sonnet-20241022 \
    --instances results/verified.jsonl --out data/submissions       # --limit 100 for a pilot, --trajs for trajectories
keel eval submission data/submissions/20250807_openhands_gpt5 data/submissions/<other> \
    --instances results/verified.jsonl --out-dir results/agents
```

`fetch` downloads, per submission, `results/results.json`, `README.md`, `metadata.yaml` from
swe-bench/experiments and `logs/<instance>/{patch.diff,report.json}` (plus `trajs/` with `--trajs`)
from the public S3 bucket, or from the submitter's own repository when `metadata.yaml` names one. No
credentials, no experiments clone, no boto3; downloads retry and resume. It warns if the README's
resolved count differs from `results.json`.

`orig` = not in the `resolved` list. Instances the agent produced no patch for (`no_generation`) are
kept as gated, wrong patches; instances you did not download (a `--limit` pilot) are left out.
Add `--judge muse-spark` to `submission` for the LLM-judge signal (one call per patch).

The fallback, if you already have a folder from the experiments repository's own
`python -m analysis.download_logs`: point `submission` at it; `--preds <file>` if predictions sit elsewhere.

## Step 2 · Oracle 2: strengthened-test labels (Docker)

The `fetch` output folder is exactly what the SWE-ABS harness reads
(`<root>/<submission>/logs/<instance>/{report.json,patch.diff}`). Two ways to run the harness.

### Route A (recommended): the keel-oracle2 image

Needs only Docker with an x86-64 Linux engine: a Linux machine, or Docker Desktop (WSL 2 backend) on
Windows or an Intel Mac, with about 16 GB memory and 120+ GB disk for Docker (Docker Desktop:
Settings → Resources). No SWE-ABS or conda install on the host.

```bash
keel eval docker-check                              # engine running? Linux, x86-64, memory, disk, image
keel eval abs-docker-build                          # builds keel-oracle2 (SWE-ABS pinned + Keel), once
keel eval abs-run --docker --models-root data/submissions --models 20250807_openhands_gpt5 \
    --run-id pilot --pilot 20                       # 20 instances first
keel eval abs-run --docker --models-root data/submissions --models 20250807_openhands_gpt5,<other> \
    --run-id full
keel eval abs-labels results/agents/20250807_openhands_gpt5.jsonl --results-dir results/sweabs --run-id full
```

The harness container gets the host's Docker socket and starts the SWE-bench instance containers next
to itself on the host; patches and tests go into them through the Docker API, so only the submissions
folder (read-only), `results/sweabs` and a Hugging Face cache volume are mounted. Output:
`results/sweabs/eval_agent/<run_id>/final_results.json`. On Linux the files are handed back to your
user. `--dry-run` prints the `docker run` command without starting anything. The image pins SWE-ABS to
commit 26b4c73 (2026-10-26); `abs-docker-build --sweabs-ref <sha>` builds another.

### Route B: SWE-ABS installed on a Linux machine

```bash
git clone https://github.com/OpenAgentEval/SWE-ABS && cd SWE-ABS
conda create -n swe-abs python=3.11 -y && conda activate swe-abs && bash setup.sh
pip install -e /path/to/keel_01
keel eval abs-run --models-root data/submissions --models 20250807_openhands_gpt5 \
    --run-id full --sweabs-dir ~/SWE-ABS
keel eval abs-labels results/agents/20250807_openhands_gpt5.jsonl --sweabs-dir ~/SWE-ABS --run-id full
```

Either way: SWE-ABS evaluates only instances the original run resolved; the others go to its
`init_fail` and stay wrong in Keel. `abs-labels` reports `flipped_to_wrong` (silent failures caught by
the stronger tests), `abs_pass_but_orig_fail` (should be about 0: an environment check), `missing`,
and SWE-ABS errors. If generator names differ from the SWE-ABS model keys, pass `--key-map map.json`.

## Step 3 · Oracle 3: human audit

```bash
keel eval audit-export results/agents/*.jsonl --instances results/verified.jsonl --out results/audit
```

This writes `items/<id>.md` (issue, patch, reference patch, checklist), `rater1.csv` (every item),
`rater2.csv` (about 30%, for agreement) and a private `key.csv`. Give each rater `items/` and only
their own sheet; they fill `verdict` with `correct`, `wrong` or `unsure` without seeing each other's.
Only audit-pool patches (a fixed 25% of issues, chosen by hash) that pass an oracle are listed;
patches the tests already reject count as wrong. Re-running the export keeps verdicts already entered.

```bash
keel eval audit-status --audit results/audit      # progress, kappa; writes adjudication.csv for disagreements
# a third person fills `final` in adjudication.csv
keel eval audit-import results/agents/<submission>.jsonl --audit results/audit   # once per agent file
keel eval labels-status results/agents/*.jsonl    # labels per oracle; which variants can run
```

`unsure` counts as wrong unless `--unsure-as-missing`. Do not change `--audit-frac` after exporting.

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

* A Docker engine: `abs-run --docker` needs Docker with an x86-64 Linux engine and enough disk.
* The human verdicts.
* Variant C (prediction-powered inference, Einbinder et al., TPAMI 2025) is not implemented.
