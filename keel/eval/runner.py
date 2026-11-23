"""Repeated random splits: fit the scorer, calibrate, and measure risk on held-out bugs."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .. import conformal
from ..baselines import BASELINES
from ..features import FEATURE_NAMES
from ..scoring import Scorer
from .records import ORACLES, has_oracle
from .records import label as rec_label

DEFAULT_ALPHAS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

FEATURE_GROUPS = {
    "adversarial_tests": ["attack_survival", "attack_minority"],
    "self_consistency": ["exact_agreement", "behavior_agreement", "mean_jaccard", "cross_model_agreement",
                         "consensus_score"],
    "diff_size_locality": ["log_diff_size", "n_files", "n_hunks", "frac_files_in_traceback", "log_line_distance"],
    "static_analysis": ["static_delta"],
    "open_weights_uncertainty": ["has_logprob", "mean_logprob", "min_logprob", "log_reasoning_tokens"],
    "self_confidence": ["self_confidence"],
}


@dataclass
class SplitResult:
    cal: list[tuple[float, bool]]
    test: list[dict]  # selected candidate per test bug: {"score", "wrong", "gates", "features", "labels"}
    grid: list[float] | None = None  # LTT grid from the fit split (independent of calibration)
    cal_sel: list[dict] | None = None
    fit_sel: list[dict] | None = None

    def ordered_grid(self, oracle: str, delta: float, n_audit: int = 0, delta2: float = 0.0) -> list[float]:
        """The fit-split-ordered grid for fixed-sequence LTT (see conformal.order_grid)."""
        fs = [x["score"] for x in self.fit_sel or []]
        fw = [bool(x["labels"][oracle]) if x["labels"][oracle] is not None else False for x in self.fit_sel or []]
        return conformal.order_grid(self.grid or conformal.LTT_GRID, fs, fw, len(self.cal), delta, n_audit, delta2)


def score_and_select(records: list[dict], scorer: Scorer) -> list[dict]:
    """Per bug: score every candidate, return the selected one (same rule as the pipeline).

    "wrong" is the primary (held-out test) label; "labels" holds the selected patch's label under
    every oracle (None where that oracle has no label).
    """
    out = []
    flat = [c for r in records for c in r["candidates"]]
    all_scores = scorer.score_many([c["features"] for c in flat], [bool(c["gates"]) for c in flat])
    pos = 0
    for r in records:
        best = None
        for c in r["candidates"]:
            s = all_scores[pos]
            pos += 1
            key = (s, -c["features"].get("behavior_agreement", 0.0), c["features"].get("log_diff_size", 0.0),
                   c["candidate_id"])
            if best is None or key < best[0]:
                best = (key, c, s)
        _, c, s = best
        out.append({"task_id": r["task_id"], "score": s, "wrong": bool(c["wrong"]), "gates": c["gates"],
                    "features": c["features"], "generator": c["generator"],
                    "candidate_id": c["candidate_id"],
                    "labels": {o: rec_label(c, o) for o in ORACLES},
                    "cost_usd": sum(x.get("cost_usd", 0.0) for x in r["candidates"])})
    return out


def fit_scorer(records: list[dict], features: list[str] | None = None, label: str = "orig") -> Scorer:
    """Fit the scorer on non-gated candidates; `label` picks the oracle used as training target."""
    feats = features or list(FEATURE_NAMES)
    X, y = [], []
    for r in records:
        for c in r["candidates"]:
            lab = rec_label(c, label)
            if not c["gates"] and lab is not None:
                X.append(c["features"])
                y.append(0 if lab else 1)
    scorer = Scorer(feature_names=feats, weights=[0.0] * len(feats), means=[0.0] * len(feats),
                    stds=[1.0] * len(feats))
    return scorer.fit(X, y)


def split(records: list[dict], rng: random.Random, fracs=(0.3, 0.35)) -> tuple[list, list, list]:
    idx = list(range(len(records)))
    rng.shuffle(idx)
    a = int(len(idx) * fracs[0])
    b = a + int(len(idx) * fracs[1])
    pick = lambda ids: [records[i] for i in ids]  # noqa: E731
    return pick(idx[:a]), pick(idx[a:b]), pick(idx[b:])


def run_split(records: list[dict], rng: random.Random, features: list[str] | None = None,
              fit_label: str = "orig", cal_label: str = "orig") -> SplitResult:
    fit, cal, test = split(records, rng)
    scorer = fit_scorer(fit, features, fit_label)
    fit_sel = score_and_select(fit, scorer)
    grid = conformal.grid_from_scores([x["score"] for x in fit_sel])
    cal_sel = score_and_select(cal, scorer)
    return SplitResult(cal=[(x["score"], bool(x["labels"][cal_label])) for x in cal_sel],
                       test=score_and_select(test, scorer), grid=grid, cal_sel=cal_sel, fit_sel=fit_sel)


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))


def auroc(scores: list[float], wrong: list[bool]) -> float:
    """P(score of a wrong patch > score of a correct one): how well the score separates them."""
    pos = [s for s, w in zip(scores, wrong) if w]
    neg = [s for s, w in zip(scores, wrong) if not w]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


class Accumulator:
    """Per-method, per-alpha risk / false-merge rate / merge rate and violation frequencies."""

    def __init__(self) -> None:
        self.acc: dict[str, dict[float, dict[str, list[float]]]] = {}

    def add(self, method: str, alpha: float, merged: list[bool], wrong: list[bool]) -> None:
        n = len(wrong)
        k = sum(merged)
        w = sum(1 for m, bad in zip(merged, wrong) if m and bad)
        slot = self.acc.setdefault(method, {}).setdefault(alpha, {"risk": [], "fdr": [], "merge": [], "viol_risk": [],
                                                                  "viol_fdr": []})
        slot["risk"].append(w / n if n else 0.0)
        slot["fdr"].append(w / k if k else 0.0)
        slot["merge"].append(k / n if n else 0.0)
        slot["viol_risk"].append(float(n > 0 and w / n > alpha))
        slot["viol_fdr"].append(float(k > 0 and w / k > alpha))

    def summary(self, alphas: list[float]) -> dict:
        out = {}
        for method, by_alpha in self.acc.items():
            out[method] = []
            for alpha in alphas:
                s = by_alpha.get(alpha)
                if not s:
                    continue
                reps = len(s["risk"])
                rm, rsd = _mean_sd(s["risk"])
                fm, fsd = _mean_sd(s["fdr"])
                mm, msd = _mean_sd(s["merge"])
                out[method].append({"alpha": alpha, "risk": rm, "risk_sd": rsd, "fdr": fm, "fdr_sd": fsd,
                                    "merge": mm, "merge_sd": msd, "viol_risk": sum(s["viol_risk"]) / reps,
                                    "viol_fdr": sum(s["viol_fdr"]) / reps})
        return out


def _require(records: list[dict], oracle: str, what: str) -> None:
    if not has_oracle(records, oracle):
        raise ValueError(f"{what} oracle {oracle!r} is missing on some candidates; add those labels first "
                         f"(keel eval labels ...) or pick another oracle")


def evaluate(records: list[dict], alphas: list[float] | None = None, n_splits: int = 200, delta: float = 0.1,
             seed: int = 0, features: list[str] | None = None, cal_label: str = "orig",
             eval_label: str = "orig") -> dict:
    """Risk, false-merge rate and merge rate of Keel (CRC, LTT) and the baselines, over random splits.

    The scorer is fit and the threshold calibrated on `cal_label`; realised error is measured on
    the test split with `eval_label`. Calibrating on "orig" and evaluating on "abs" / "human" / "true"
    measures how much an imperfect oracle inflates the guarantee (thesis RQ1).
    """
    alphas = alphas or DEFAULT_ALPHAS
    _require(records, cal_label, "calibration")
    _require(records, eval_label, "evaluation")
    rng = random.Random(seed)
    acc = Accumulator()
    aucs: list[float] = []

    for _ in range(n_splits):
        sr = run_split(records, rng, features, fit_label=cal_label, cal_label=cal_label)
        cs, cw = [s for s, _ in sr.cal], [w for _, w in sr.cal]
        tw = [bool(t["labels"][eval_label]) for t in sr.test]
        aucs.append(auroc([t["score"] for t in sr.test], tw))
        grids = {"crc": None, "ltt": sr.ordered_grid(cal_label, delta), "ltt_bonferroni": sr.grid}
        for alpha in alphas:
            for method in ("crc", "ltt", "ltt_bonferroni"):
                thr = conformal.threshold(method, cs, cw, alpha, delta, grids[method])
                acc.add(f"keel_{method}", alpha, [t["score"] <= thr and t["score"] < 1 for t in sr.test], tw)
            for name, rule in BASELINES.items():
                acc.add(name, alpha, [rule(t, alpha) for t in sr.test], tw)
            acc.add("oracle", alpha, [not w and not t["gates"] for w, t in zip(tw, sr.test)], tw)

    finite = [a for a in aucs if not math.isnan(a)]
    return {"alphas": alphas, "n_splits": n_splits, "delta": delta, "n_bugs": len(records),
            "cal_label": cal_label, "eval_label": eval_label,
            "auroc": _mean_sd(finite)[0] if finite else None, "methods": acc.summary(alphas)}


def slider_data(records: list[dict], seed: int = 0) -> dict:
    """One fixed split for the interactive slider: calibration + test points, and the fitted scorer."""
    rng = random.Random(seed)
    sr = run_split(records, rng)
    return {"cal": [[round(s, 6), int(w)] for s, w in sr.cal],
            "test": [[round(t["score"], 6), int(t["wrong"]), int(bool(t["gates"])),
                      round(t["features"].get("self_confidence", 0.5), 4)] for t in sr.test]}


def model_comparison(records: list[dict], alpha: float = 0.1, n_splits: int = 100, seed: int = 0) -> list[dict]:
    from .records import restrict

    gens = sorted({c["generator"] for r in records for c in r["candidates"]})
    configs = [[g] for g in gens] + ([gens] if len(gens) > 1 else [])
    rows = []
    for cfg in configs:
        sub = restrict(records, set(cfg))
        if len(sub) < 20:
            continue
        ev = evaluate(sub, alphas=[alpha], n_splits=n_splits, seed=seed)
        keel = ev["methods"]["keel_crc"][0]
        solved = sum(1 for r in sub if any(not c["wrong"] and not c["gates"] for c in r["candidates"])) / len(sub)
        cost = sum(c.get("cost_usd", 0.0) for r in sub for c in r["candidates"]) / len(sub)
        rows.append({"models": "+".join(cfg), "solve_at_k": solved, "auroc": ev["auroc"],
                     "merge_rate": keel["merge"], "risk": keel["risk"], "cost_per_bug": cost,
                     "cost_per_auto_merge": cost / keel["merge"] if keel["merge"] > 0 else None})
    return rows


def ablations(records: list[dict], alpha: float = 0.1, n_splits: int = 100, seed: int = 0) -> list[dict]:
    rows = []
    full = evaluate(records, alphas=[alpha], n_splits=n_splits, seed=seed)
    base = full["methods"]["keel_crc"][0]
    rows.append({"removed": "(none)", "merge_rate": base["merge"], "risk": base["risk"], "auroc": full["auroc"]})
    for group, feats in FEATURE_GROUPS.items():
        keep = [f for f in FEATURE_NAMES if f not in feats]
        ev = evaluate(records, alphas=[alpha], n_splits=n_splits, seed=seed, features=keep)
        m = ev["methods"]["keel_crc"][0]
        rows.append({"removed": group, "merge_rate": m["merge"], "risk": m["risk"], "auroc": ev["auroc"]})
    return rows
