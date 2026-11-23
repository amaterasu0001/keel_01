<p align="center">
  <img src="web/public/logo.svg" width="72" alt="Keel logo">
</p>

<h1 align="center">Keel</h1>
<p align="center"><b>A code-repair agent that knows when it should not be trusted.</b></p>
<p align="center">
  <a href="https://keel.boraqs.com">keel.boraqs.com</a> ·
  <a href="https://www.youtube.com/watch?v=I77PAVkJDCI"><b>Demo Video</b></a> ·
  <a href="https://feedofmri.github.io/keel/">Preview (GitHub Pages)</a> ·
  <a href="https://feedofmri.github.io/keel/results">Results</a> ·
  <a href="#install-as-a-github-action">Install</a> ·
  <a href="HACKATHON.md">What we built at the hackathon</a>
</p>

---

Coding agents now produce plausible patches faster than teams can review them. The bottleneck is no
longer *generating* fixes but *deciding which ones are safe to merge without a human*.

Keel uses **Meta's Muse models** to generate candidate patches for failing tests. Then:

1. Muse Spark **writes adversarial tests** aimed at where the candidates disagree.
2. Keel scores each patch with a **code-aware nonconformity score**: test outcomes, adversarial survival,
   diff size and locality, static-analysis deltas, self-consistency across samples and across models,
   and token-level uncertainty from the open-weights model.
3. A **distribution-free calibration** layer (Learn-then-Test or conformal risk control), calibrated on
   held-out bug fixes, turns the score into a decision with a guarantee at a risk level you choose
   (say 10%). With LTT: with probability ≥ 1 − δ, at most 10% of auto-merged patches are wrong.
4. Keel opens a pull request labelled **AUTO-MERGE** or **NEEDS REVIEW**, with an explanation.

> Built for the Meta Global AI Developer Hackathon 2026 (*Push the Frontier*, software). All code in
> this repository was written during the event. See [`HACKATHON.md`](HACKATHON.md).

<p align="center">
  <a href="https://www.youtube.com/watch?v=I77PAVkJDCI">
    <img src="https://img.youtube.com/vi/I77PAVkJDCI/maxresdefault.jpg" width="700" alt="Watch the Keel Demo Video">
  </a>
  <br>
  <b><a href="https://www.youtube.com/watch?v=I77PAVkJDCI">▶️ Watch the 3-Minute Demo Video & Live Walkthrough</a></b>
</p>


## How Meta's models are used

| Model | Access | Role |
|-------|--------|------|
| **Muse Spark** | Meta Model API | Main patch generator; writes the adversarial tests; optional plain-English PR explanation |
| **Muse Spark Open** (`meta-models/muse-spark-open-1026`) | Open weights via Baseten (or any OpenAI-compatible partner) | Second generator for diversity; **token logprobs and reasoning length** become uncertainty features, which only open weights expose |
| **Muse Glimmer** | Open weights via a partner | Optional cheap triage that sets the sample budget; cost baseline in the evaluation |

Every model goes through one OpenAI-compatible client, so changing provider is a configuration change.

## The guarantee

Calibration data: *n* held-out bugs, each with the risk score *sᵢ* of Keel's selected patch and a label
*wrongᵢ* from **hidden tests the agent never saw**. Keel auto-merges when *s ≤ λ̂*.

* **CRC (product default):** λ̂ = max{λ : (Σᵢ wrongᵢ·1[sᵢ ≤ λ] + 1)/(n + 1) ≤ α}. Then
  **E[1{auto-merged and wrong}] ≤ α**: on average, at most α of *all bugs* get a wrong patch
  auto-merged (Angelopoulos et al., *Conformal Risk Control*, 2022). This is a bound per bug, not
  per merge: when few patches are merged, the share of merged patches that are wrong can exceed α.
* **LTT (the thesis method):** each threshold on a grid fixed before calibration (from the scorer's
  fit split) is tested with an exact binomial test, and the family-wise error is controlled with
  **fixed-sequence testing** (most conservative threshold first, stop at the first failure). Then
  **with probability ≥ 1 − δ, at most α of auto-merged patches are wrong** (the false-merge rate;
  Angelopoulos et al., *Learn then Test*, 2021). Bonferroni is kept as `ltt_bonferroni` for comparison;
  it needs about twice as many calibration bugs (49 instead of 22 error-free merges at α = δ = 10%).
* **Three tiers:** merge / review / reject. The reject threshold is certified jointly (δ split in half):
  at most β (`beta` in `.keel.toml`, default 10%) of the patches Keel marks *likely wrong* are actually
  correct. The pull-request label stays AUTO-MERGE / NEEDS REVIEW; likely-wrong patches say so in the PR.

"Wrong" means failing the held-out tests. Tests can miss wrong patches, so these guarantees are only
as good as the tests. `keel/conformal_noisy.py` implements a noise-corrected LTT that also bounds the
*true* false-merge rate using a small human audit (see [the thesis workflow](docs/thesis-gap1.md)).

Assumptions are stated in every PR. Calibration and deployment bugs must be exchangeable, so Keel
fingerprints its models, prompts and scorer and refuses to auto-merge on a stale calibration. With no
calibration at all, Keel never auto-merges.

`tests/test_conformal.py` and `tests/test_thesis.py` check the guarantees by simulation on every CI run.

## The Keel platform

At [keel.boraqs.com/app](https://keel.boraqs.com/app/), sign in with GitHub, pick a repository and
set a risk level. Keel stores your Meta model keys as encrypted repository secrets and opens one setup
pull request. After you merge it, every run shows up on the repository's dashboard. The code is in
[`platform/`](platform/) (Node.js, Fastify, PostgreSQL), and deployment is described in
[`docs/cpanel-setup.md`](docs/cpanel-setup.md). The web app is React in [`web/`](web/); the GitHub Pages
copy is the same build in demo mode with sample data.

## Quick start (local)

```bash
pip install git+https://github.com/feedofmri/keel
keel doctor                      # which Meta model endpoints are configured
keel repair --repo path/to/repo  # discover failing tests, repair, print the PR body
```

Offline demo with canned model responses (no API keys):

```bash
keel repair --repo examples/demo_repo --cache "" \
  --models scripted:muse-spark:examples/responses/muse-spark.json,scripted:muse-spark-open:examples/responses/muse-spark-open.json \
  --attacker scripted:attacker:examples/responses/attacker.json
```

## Install as a GitHub Action

Copy [`examples/workflows/keel.yml`](examples/workflows/keel.yml) to `.github/workflows/keel.yml`
and add your model keys as repository secrets. Keel runs when CI fails, or when someone with write
access comments `/keel fix`. Configure it with [`.keel.toml`](examples/keel-demo/.keel.toml).
Secrets are scrubbed from the environment of any code a model wrote, and edits to tests or CI are
hard-blocked.

## Evaluation

```bash
keel eval quixbugs  --models muse-spark,muse-spark-open --k 5    # 40 programs, visible/hidden split
keel eval swebench  --n 100 --stage all                          # SWE-bench Lite, official harness
keel eval analyze   results/quixbugs/records.jsonl --out web/public/data/results.json
keel calibrate      results/quixbugs/records.jsonl --out calibration.json
```

Thesis experiments (certified acceptance under imperfect test oracles) use published agent patches,
strengthened tests and a human audit; the full step-by-step workflow is in
[`docs/thesis-gap1.md`](docs/thesis-gap1.md):

```bash
keel eval instances    --out results/verified.jsonl
keel eval fetch        <submission> --instances results/verified.jsonl --out data/submissions   # Oracle 1 input
keel eval submission   data/submissions/<submission> --instances results/verified.jsonl --out-dir results/agents
keel eval abs-run      --models <submission> --run-id full --sweabs-dir ~/SWE-ABS              # Oracle 2 (Linux + Docker)
keel eval abs-labels   results/agents/<submission>.jsonl --sweabs-dir ~/SWE-ABS --run-id full
keel eval audit-export results/agents/<submission>.jsonl --instances results/verified.jsonl --out results/audit
keel eval audit-status --audit results/audit                                                     # Oracle 3
keel eval audit-import results/agents/<submission>.jsonl --audit results/audit
keel eval labels-status results/agents/<submission>.jsonl
keel eval thesis       results/agents/<submission>.jsonl --out-dir results/thesis
```

Both benchmarks also run from the Actions tab (`Benchmark – QuixBugs`, `Benchmark – SWE-bench Lite`).
Results feed the [results page](https://feedofmri.github.io/keel/results). It compares Keel with
merging whenever tests pass, with model self-confidence ≥ 1−α, and with Keel's own score without
calibration. It also shows per-model cost and feature ablations.

**Protocol.** Keel only sees half of the failing tests. A patch is *correct* iff it passes all hidden
tests (for SWE-bench: every FAIL_TO_PASS and PASS_TO_PASS test, in the official Docker harness). For
QuixBugs, a check confirms that the reference fix passes and the buggy program fails the hidden tests for
all 40 programs. Bugs are split into fit / calibration / test sets, repeated over 200 random splits.

## Tests

```bash
pytest -q && ruff check keel tests                           # engine and Action (includes the run-report contract)
cd platform && TEST_DATABASE_URL=postgres://… npm test       # API, plus end to end against a fake GitHub
E2E_BROWSER=1 TEST_DATABASE_URL=postgres://… npx vitest run test/browser.e2e.test.ts   # live web build in Chromium
cd web && npm run build                                      # typecheck and build the web app
```

CI runs all of them, on PostgreSQL 12 and 16.

## Repository map

| Path | What |
|------|------|
| `keel/pipeline.py` | Baseline → generate → attack → evaluate → score → decide |
| `keel/generators/` | OpenAI-compatible client with Meta model profiles; scripted generator |
| `keel/attack.py` | Adversarial test generation |
| `keel/features.py`, `keel/scoring.py` | Code-aware features, gates, nonconformity score |
| `keel/conformal.py`, `keel/calibration.py` | CRC and Learn-then-Test, calibration artefact |
| `keel/github/` | The GitHub Action entry point |
| `keel/eval/` | QuixBugs and SWE-bench adapters, split runner, simulation |
| `web/` | The web app (React): public pages, sign-in, and the app screens |
| `platform/` | The Keel platform API (Node.js, Fastify, PostgreSQL), which also serves `web/` |
| `examples/` | Demo repo, canned responses, workflow, live-demo template |

## License

MIT. QuixBugs and SWE-bench are downloaded at run time under their own MIT licenses and are not
redistributed.
