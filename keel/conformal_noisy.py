"""Noise-corrected Learn-then-Test: false-merge guarantees when the test oracle itself is wrong.

Setting. Each patch has a score s (lower = more trustworthy), an oracle label W_o (1 = the oracle,
e.g. the held-out tests, says the patch is wrong) and a true label W (1 = the patch is actually
wrong), known only for a small human-audited sample. Oracle noise for code patches is mostly
one-sided: weak tests let wrong patches pass (W = 1, W_o = 0). Since a wrong patch either fails
the oracle or slips past it,

    FMR_true(lam) = P(W = 1 | s <= lam)
                 <= P(W_o = 1 | s <= lam) + eps(lam),   eps(lam) = P(W = 1, W_o = 0 | s <= lam).

Correct patches that the oracle rejects (W = 0, W_o = 1) only make this bound looser, so they need
no correction.

Procedure (`noise_corrected_threshold`). On a grid fixed in advance, from the most conservative
threshold up: U1 = Clopper-Pearson bound (level delta1) on P(W_o = 1 | s <= lam) from the
calibration set; U2 = Clopper-Pearson bound (level delta2) on eps(lam) from an independent audit
set; accept lam while U1 + U2 <= alpha, stop at the first failure (fixed-sequence testing).

Guarantee. If the calibration patches and a deployment patch are exchangeable, the audit set is an
independent sample from the same distribution, and the human labels are correct, then
P(FMR_true(lam_hat) <= alpha) >= 1 - delta1 - delta2. Each step is a level-(delta1 + delta2) test
by a union bound; fixed-sequence testing then controls the family-wise error without monotonicity.

Variants used in the thesis:
  A  oracle = original held-out tests; audit measures patches that pass them but are wrong.
  B  oracle = strengthened tests (SWE-ABS); audit measures what even those miss ("oracle ladder").
  plain  ordinary LTT on the oracle labels, i.e. U2 = 0 (what Keel did before; optimistic).
"""

from __future__ import annotations

from dataclasses import dataclass

from .conformal import GATED, LTT_GRID, NEVER, binom_ucb


@dataclass
class StepStats:
    threshold: float
    n_cal: int
    n_cal_wrong: int
    u1: float
    n_audit: int
    n_audit_missed: int
    u2: float
    accepted: bool


def _count(scores, flags, lam):
    k = w = 0
    for s, f in zip(scores, flags):
        if s <= lam and s < GATED:
            k += 1
            w += bool(f)
    return k, w


def missed_flags(oracle_wrong: list[bool], true_wrong: list[bool | None]) -> list[bool]:
    """1{W = 1 and W_o = 0} per audit patch. Patches the oracle already rejects need no human label."""
    out = []
    for o, t in zip(oracle_wrong, true_wrong):
        if o:
            out.append(False)
        elif t is None:
            raise ValueError("an audit patch that passes the oracle has no human label; label it or drop its issue")
        else:
            out.append(bool(t))
    return out


def noise_corrected_scan(cal_scores: list[float], cal_oracle_wrong: list[bool],
                         audit_scores: list[float], audit_missed: list[bool],
                         alpha: float, delta1: float = 0.05, delta2: float = 0.05,
                         grid: list[float] | None = None, use_audit: bool = True) -> list[StepStats]:
    """Every step of the fixed-sequence scan (for explanations and plots)."""
    steps = []
    for lam in sorted(grid or LTT_GRID):
        k1, w1 = _count(cal_scores, cal_oracle_wrong, lam)
        u1 = binom_ucb(w1, k1, delta1)
        if use_audit:
            k2, w2 = _count(audit_scores, audit_missed, lam)
            u2 = binom_ucb(w2, k2, delta2)
        else:
            k2 = w2 = 0
            u2 = 0.0
        ok = u1 + u2 <= alpha
        steps.append(StepStats(lam, k1, w1, u1, k2, w2, u2, ok))
        if not ok:
            break
    return steps


def noise_corrected_threshold(cal_scores: list[float], cal_oracle_wrong: list[bool],
                              audit_scores: list[float], audit_missed: list[bool],
                              alpha: float, delta1: float = 0.05, delta2: float = 0.05,
                              grid: list[float] | None = None) -> float:
    """Largest grid threshold with P(true false-merge rate <= alpha) >= 1 - delta1 - delta2."""
    best = NEVER
    for st in noise_corrected_scan(cal_scores, cal_oracle_wrong, audit_scores, audit_missed, alpha,
                                   delta1, delta2, grid):
        if st.accepted:
            best = st.threshold
    return best


def plain_threshold(cal_scores: list[float], cal_oracle_wrong: list[bool], alpha: float, delta: float = 0.1,
                    grid: list[float] | None = None) -> float:
    """Ordinary fixed-sequence LTT on oracle labels (U2 = 0). Equivalent to conformal.ltt_threshold."""
    best = NEVER
    for st in noise_corrected_scan(cal_scores, cal_oracle_wrong, [], [], alpha, delta, 0.0, grid, use_audit=False):
        if st.accepted:
            best = st.threshold
    return best


def guarantee_text(variant: str, alpha: float, delta1: float, delta2: float, n_cal: int, n_audit: int) -> str:
    oracle = {"A": "the held-out tests", "B": "strengthened tests"}.get(variant, "the oracle")
    return (f"Noise-corrected Learn-then-Test (variant {variant}) on {n_cal} calibration bugs and {n_audit} "
            f"human-audited bugs: with probability at least {1 - delta1 - delta2:.0%}, at most {alpha:.0%} of "
            f"auto-merged patches are actually wrong, even counting wrong patches that pass {oracle}.")
