# Demo Video Walkthrough

**YouTube Link:** [https://www.youtube.com/watch?v=I77PAVkJDCI](https://www.youtube.com/watch?v=I77PAVkJDCI)

*Core Hackathon 3-Minute Evaluation:* **0:00 – 3:00**

## Video Structure & Walkthrough

| Time | Screen / Context | Voice-over & Technical Focus |
|------|------------------|------------------------------|
| 0:00–0:35 | **Live Platform:** [keel.boraqs.com](https://keel.boraqs.com) | The AI code review bottleneck: 30–60% of agent PRs fail or regress. Introducing Keel: autonomous code repair with a distribution-free error bound. |
| 0:35–1:05 | **GitHub Repo:** [feedofmri/keel](https://github.com/feedofmri/keel) | Multi-model architecture: Muse Spark (Meta Model API) and Muse Spark Open (Baseten) propose fixes. Muse Spark writes adversarial tests. Token logprobs provide open-weights uncertainty. |
| 1:05–1:40 | **Live GitHub PR #1:** keel-demo | Clean bug fix: CI fails, Keel triggers on GitHub runner, opens PR with green **AUTO-MERGE** label. Nonconformity score  \le \hat{\lambda}$. |
| 1:40–2:10 | **Live GitHub PR #2:** keel-demo | Deceptive bug: passes visible tests, but fails Muse Spark adversarial attack. High uncertainty flags PR with yellow **NEEDS REVIEW** label and explanation. |
| 2:10–2:40 | **Interactive Dashboard:** /results | Real-time **Risk Slider ($\\alpha$)**: dragging $\\alpha$ from 5% to 20% updates thresholds dynamically. Empirical error on SWE-bench Lite and QuixBugs remains strictly below the bound. |
| 2:40–3:00 | **Provenance & Meta Lineage** | Built 100% during the hackathon (Oct 26 – Nov 4). Assured LLM software engineering with Meta Muse models. |
| 3:00–5:00 | **Extended Deep Dive** | Deep dive into feature extraction (20 signals), token logprob uncertainty, Fastify/PostgreSQL architecture, and calibration artifacts. |
