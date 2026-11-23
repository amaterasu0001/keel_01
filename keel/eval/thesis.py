"""Thesis experiments: certified patch acceptance under imperfect test oracles.

RQ1  inflation    calibrate on the held-out tests ("orig"), measure the realised false-merge rate
                  against a better oracle ("abs", "human", or "true" in simulation)
RQ3  correction   noise-corrected LTT (variants A and B) versus plain LTT and the baselines
RQ4  audit budget how the merge rate of the corrected gate grows with the number of audited bugs

Splits. A fixed audit pool (chosen by a hash of the task id, so the same bugs are audited in every
repetition and humans only label those) is set aside first. The remaining bugs are split at random,
per repetition, into fit / calibration / test. Within a repetition the audit set may be subsampled
to a budget m (RQ4).

Human labels are needed only for audit-pool patches that the oracle passes: a patch the oracle
already calls wrong counts as wrong without looking at it (see conformal_noisy.missed_flags).
"""

from __future__ import annotations

import hashlib
import math
import random

from .. import conformal, conformal_noisy
from ..baselines import BASELINES
from .records import label as rec_label
from .runner import Accumulator, auroc, fit_scorer, score_and_select, split

VARIANTS = {
    # name: (calibration oracle, audit compares against this oracle, uses audit)
    "plain": ("orig", None, False),
    "plain_abs": ("abs", None, False),
    "A": ("orig", "orig", True),
    "B": ("abs", "abs", True),
}


def in_audit_pool(task_id: str, frac: float, salt: str = "keel-audit") -> bool:
    h = int(hashlib.sha256(f"{salt}:{task_id}".encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < frac


def partition(records: list[dict], audit_frac: float = 0.25, salt: str = "keel-audit") -> tuple[list, list]:
    """(audit pool, rest)."""
    pool = [r for r in records if in_audit_pool(r["task_id"], audit_frac, salt)]
    rest = [r for r in records if not in_audit_pool(r["task_id"], audit_frac, salt)]
    return pool, rest


def audit_requests(records: list[dict], audit_frac: float = 0.25, oracles: tuple[str, ...] = ("orig", "abs"),
                   salt: str = "keel-audit") -> list[dict]:
    """Candidates in the audit pool that need a human label: those that pass at least one oracle used.

    Every candidate of a pool bug is listed (not only the one a particular scorer would select),
    because the selected candidate depends on the scorer, which changes between repetitions.
    """
    pool, _ = partition(records, audit_frac, salt)
    out = []
    for r in pool:
        for c in r["candidates"]:
            if c["gates"]:
                continue
            passes = [o for o in oracles if rec_label(c, o) is False]
            if passes:
                out.append({"task_id": r["task_id"], "candidate_id": c["candidate_id"], "generator": c["generator"],
                            "passes": passes})
    return out


def _available(variants: list[str], records: list[dict]) -> list[str]:
    have_abs = all(rec_label(c, "abs") is not None for r in records for c in r["candidates"])
    return [v for v in variants if have_abs or VARIANTS[v][0] != "abs"]


def evaluate_noisy(records: list[dict], alphas: list[float] | None = None, n_splits: int = 200,
                   delta: float = 0.1, seed: int = 0, features: list[str] | None = None,
                   eval_label: str = "best", audit_frac: float = 0.25, audit_budget: int | None = None,
                   variants: list[str] | None = None, salt: str = "keel-audit") -> dict:
    """RQ1 + RQ3 (+ RQ4 with audit_budget): violation and merge rates against `eval_label`.

    delta is the total error budget; corrected variants split it evenly between the calibration
    bound and the audit bound (delta1 = delta2 = delta / 2).
    """
    alphas = alphas or [0.05, 0.1, 0.15, 0.2, 0.3]
    variants = _available(variants or list(VARIANTS), records)
    pool, rest = partition(records, audit_frac, salt)
    if len(rest) < 10:
        raise ValueError("too few bugs outside the audit pool")
    rng = random.Random(seed)
    acc = Accumulator()
    aucs: list[float] = []
    n_audit_used: list[int] = []
    skipped = 0

    for _ in range(n_splits):
        fit, cal, test = split(rest, rng, fracs=(0.34, 0.33))
        # The scorer is trained on the held-out-test labels in every variant, so variants differ only
        # in how they calibrate. (A team without strengthened tests could not train on them either.)
        scorer = fit_scorer(fit, features, "orig")
        fit_sel = score_and_select(fit, scorer)
        grid = conformal.grid_from_scores([x["score"] for x in fit_sel])
        cal_sel = score_and_select(cal, scorer)
        test_sel = [t for t in score_and_select(test, scorer) if t["labels"][eval_label] is not None]
        if not test_sel:
            skipped += 1
            continue
        audit = list(pool)
        if audit_budget is not None and audit_budget < len(audit):
            audit = rng.sample(audit, audit_budget)
        audit_sel = score_and_select(audit, scorer)
        n_audit_used.append(len(audit_sel))
        tw = [bool(t["labels"][eval_label]) for t in test_sel]
        aucs.append(auroc([t["score"] for t in test_sel], tw))

        fs = [x["score"] for x in fit_sel]
        cs = [x["score"] for x in cal_sel]
        prepared = {}  # per variant: (calibration labels, ordered grid, audit missed flags or None)
        for v in variants:
            cal_oracle, audit_oracle, uses_audit = VARIANTS[v]
            cw = [bool(x["labels"][cal_oracle]) for x in cal_sel]
            fw = [bool(x["labels"][cal_oracle]) for x in fit_sel]
            if uses_audit:
                try:
                    missed = conformal_noisy.missed_flags([bool(x["labels"][audit_oracle]) for x in audit_sel],
                                                          [x["labels"]["human"] for x in audit_sel])
                except ValueError:
                    continue  # audit labels incomplete for this oracle: variant not reported
                g = conformal.order_grid(grid, fs, fw, len(cal_sel), delta / 2, len(audit_sel), delta / 2)
                prepared[v] = (cw, g, missed)
            else:
                prepared[v] = (cw, conformal.order_grid(grid, fs, fw, len(cal_sel), delta), None)
        audit_scores = [x["score"] for x in audit_sel]

        for alpha in alphas:
            for v, (cw, g, missed) in prepared.items():
                if missed is not None:
                    thr = conformal_noisy.noise_corrected_threshold(cs, cw, audit_scores, missed,
                                                                    alpha, delta / 2, delta / 2, g)
                else:
                    thr = conformal.ltt_threshold(cs, cw, alpha, delta, g)
                acc.add(v, alpha, [t["score"] <= thr and t["score"] < conformal.GATED for t in test_sel], tw)
            for name, rule in BASELINES.items():
                acc.add(name, alpha, [rule(t, alpha) for t in test_sel], tw)

    finite = [a for a in aucs if not math.isnan(a)]
    return {"alphas": alphas, "n_splits": n_splits - skipped, "delta": delta, "eval_label": eval_label,
            "n_bugs": len(records), "n_audit_pool": len(pool), "audit_budget": audit_budget,
            "mean_audit_used": sum(n_audit_used) / len(n_audit_used) if n_audit_used else 0,
            "auroc": sum(finite) / len(finite) if finite else None, "variants": variants,
            "methods": acc.summary(alphas)}


def inflation(records: list[dict], alphas: list[float] | None = None, n_splits: int = 200, delta: float = 0.1,
              seed: int = 0, features: list[str] | None = None, oracles: tuple[str, ...] = ("orig", "abs", "human",
                                                                                              "true")) -> dict:
    """RQ1: plain LTT calibrated on the held-out tests, judged by every available oracle."""
    alphas = alphas or [0.05, 0.1, 0.15, 0.2, 0.3]
    rows = {}
    for o in oracles:
        # Evaluate only on bugs whose selected patch has this oracle's label: run per oracle so a
        # partially labelled oracle (human) is measured on the patches it covers.
        ev = evaluate_noisy(records, alphas, n_splits, delta, seed, features, eval_label=o, audit_frac=0.0,
                            variants=["plain"])
        if ev["methods"].get("plain"):
            rows[o] = ev["methods"]["plain"]
    return {"alphas": alphas, "delta": delta, "by_oracle": rows}


def audit_budget_curve(records: list[dict], budgets: list[int], alpha: float = 0.1, n_splits: int = 100,
                       delta: float = 0.1, seed: int = 0, features: list[str] | None = None,
                       eval_label: str = "best", audit_frac: float = 0.25) -> list[dict]:
    """RQ4: merge rate and violation of the corrected variants as the audit budget grows."""
    out = []
    for m in budgets:
        ev = evaluate_noisy(records, [alpha], n_splits, delta, seed, features, eval_label, audit_frac, m,
                            variants=["A", "B"])
        row = {"budget": m, "audit_used": ev["mean_audit_used"]}
        for v in ("A", "B"):
            r = (ev["methods"].get(v) or [None])[0]
            if r:
                row[f"{v}_merge"] = r["merge"]
                row[f"{v}_viol"] = r["viol_fdr"]
                row[f"{v}_fdr"] = r["fdr"]
        out.append(row)
    return out
