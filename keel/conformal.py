"""Distribution-free decision thresholds.

Decision rule: auto-merge a bug's selected patch iff its score s <= lambda.
Calibration data: n held-out bugs with score s_i and wrong_i in {0, 1} (from hidden tests).

Two guarantees are offered (both assume calibration and test bugs are exchangeable):

* LTT (Learn-then-Test, Angelopoulos et al. 2021), the thesis method and the default for the
  analysis: for each lambda on a grid fixed before calibration, test
  H0: FMR(lambda) = P(wrong | s <= lambda) > alpha with an exact binomial p-value, and control the
  family-wise error with fixed-sequence testing (most conservative lambda first, stop at the first
  non-rejection). With probability >= 1 - delta over the calibration draw, the share of
  AUTO-MERGED patches that are wrong (the false-merge rate) is at most alpha.
  Bonferroni over the grid is kept as `procedure="bonferroni"` for comparison.

* CRC (conformal risk control, Angelopoulos et al. 2022): with loss
  L_i(lambda) = wrong_i * 1{s_i <= lambda} (monotone, bounded by 1), choosing
  lambda_hat = max{lambda : (sum_i L_i(lambda) + 1) / (n + 1) <= alpha}
  gives E[ 1{merged and wrong} ] <= alpha for a new bug. This bounds wrong auto-merges as a share
  of ALL bugs, in expectation. It is NOT a bound on the share of merged patches that are wrong.

Three-tier decisions (merge / review / reject) are calibrated jointly by splitting delta between a
merge threshold (false-merge rate <= alpha) and a reject threshold (share of correct patches among
the rejected <= beta), see `tier_thresholds`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

NEVER = -1.0  # threshold that merges nothing (all scores are >= 0)
GATED = 1.0   # scores of gated patches; thresholds are always below this
NO_REJECT = 2.0  # reject threshold that rejects nothing (all scores are <= 1)

MERGE, REVIEW, REJECT = "merge", "review", "reject"


@dataclass
class RuleStats:
    threshold: float
    merge_rate: float  # share of bugs auto-merged
    risk: float        # share of bugs wrongly auto-merged (CRC risk)
    fdr: float         # share of auto-merged patches that are wrong (false-merge rate; 0 if none merged)
    n_merged: int
    n_wrong_merged: int


def rule_stats(scores: list[float], wrong: list[bool], threshold: float) -> RuleStats:
    n = len(scores)
    merged = [s <= threshold and s < GATED for s in scores]
    k = sum(merged)
    w = sum(1 for m, bad in zip(merged, wrong) if m and bad)
    return RuleStats(threshold=threshold, merge_rate=k / n if n else 0.0, risk=w / n if n else 0.0,
                     fdr=w / k if k else 0.0, n_merged=k, n_wrong_merged=w)


def crc_threshold(scores: list[float], wrong: list[bool], alpha: float, bound: float = 1.0) -> float:
    """Largest threshold whose adjusted calibration risk is <= alpha (NEVER if none)."""
    n = len(scores)
    if n == 0:
        return NEVER
    pairs = sorted((s, bool(b)) for s, b in zip(scores, wrong) if s < GATED)
    best = NEVER
    if bound / (n + 1) > alpha:
        return NEVER
    loss = 0
    i = 0
    while i < len(pairs):
        s = pairs[i][0]
        # include all calibration points tied at s
        while i < len(pairs) and pairs[i][0] == s:
            loss += pairs[i][1]
            i += 1
        if (loss + bound) / (n + 1) <= alpha:
            best = s
        else:
            break  # loss is monotone in the threshold
    return best


# ---------------------------------------------------------------------------------------------
# Exact binomial tools
# ---------------------------------------------------------------------------------------------

def _log_binom_cdf(w: int, k: int, p: float) -> float:
    """log P(Bin(k, p) <= w)."""
    if w >= k:
        return 0.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return -math.inf
    terms = [math.lgamma(k + 1) - math.lgamma(i + 1) - math.lgamma(k - i + 1)
             + i * math.log(p) + (k - i) * math.log1p(-p) for i in range(w + 1)]
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def binom_pvalue(w: int, k: int, alpha: float) -> float:
    """p-value for H0: wrong-rate among k merged > alpha, having observed w wrong."""
    if k == 0:
        return 1.0
    return math.exp(_log_binom_cdf(w, k, alpha))


@lru_cache(maxsize=65536)
def binom_ucb(w: int, k: int, delta: float) -> float:
    """Clopper-Pearson upper confidence bound: P(p <= ucb) >= 1 - delta, from w successes in k.

    It is the largest p with P(Bin(k, p) <= w) >= delta, so `binom_pvalue(w, k, a) <= delta` holds
    exactly when a >= ucb (up to bisection tolerance). Returns 1.0 when k == 0.
    """
    if k == 0 or w >= k:
        return 1.0
    lo, hi = 0.0, 1.0
    log_d = math.log(delta)
    for _ in range(50):  # 2^-50 tolerance
        mid = (lo + hi) / 2
        if _log_binom_cdf(w, k, mid) > log_d:
            lo = mid
        else:
            hi = mid
    return hi


# ---------------------------------------------------------------------------------------------
# Learn-then-Test
# ---------------------------------------------------------------------------------------------

# Fallback grid, fixed in advance (never chosen from calibration data). Dense near 0 where good
# patches score. Prefer `grid_from_scores` on the scorer's FIT split: with fixed-sequence testing a
# grid point that merges no calibration patch stops the sequence, so the grid should start where
# patches actually score.
LTT_GRID = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

PROCEDURES = ("fixed_sequence", "bonferroni")


def grid_from_scores(scores: list[float], n: int = 20) -> list[float]:
    """A threshold grid from quantiles of scores that are independent of the calibration set.

    Use scores of the fit (training) split. The grid only has to be fixed before the calibration
    data is looked at; it may depend on any data disjoint from it.
    """
    xs = sorted(s for s in scores if s < GATED)
    if not xs:
        return list(LTT_GRID)
    qs = [xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] for q in [(i + 1) / n for i in range(n)]]
    grid = sorted({round(min(q, GATED - 1e-9), 12) for q in qs})
    return grid


def order_grid(grid: list[float], fit_scores: list[float], fit_wrong: list[bool], n_cal: int, delta: float,
               n_audit: int = 0, delta2: float = 0.0) -> list[float]:
    """Where the fixed sequence should start, decided on the FIT split only.

    Fixed-sequence testing stops at the first threshold it cannot certify, so starting at a very
    conservative threshold that merges only a handful of calibration patches wastes the whole
    budget. This estimates, from the fit split scaled to the calibration (and audit) sizes, the upper
    bound each threshold would get, and drops the thresholds below the most promising one. Using
    data disjoint from calibration keeps the guarantee intact (only the order is data-driven).
    """
    grid = sorted(grid)
    if not grid or not fit_scores:
        return grid
    n_fit = len(fit_scores)
    best_i, best_v = 0, math.inf
    for i, lam in enumerate(grid):
        st = rule_stats(fit_scores, fit_wrong, lam)
        k = round(st.n_merged * n_cal / n_fit)
        w = round(st.n_wrong_merged * n_cal / n_fit)
        v = binom_ucb(w, k, delta)
        if n_audit:
            v += binom_ucb(0, round(st.n_merged * n_audit / n_fit), delta2)
        if v < best_v - 1e-12:
            best_i, best_v = i, v
    return grid[best_i:]


def _ltt_scan(stats: list[tuple[float, int, int]], alpha: float, delta: float, procedure: str) -> float:
    """stats: (lambda, n_merged, n_wrong_merged) in increasing lambda."""
    if procedure not in PROCEDURES:
        raise ValueError(f"unknown LTT procedure {procedure!r}")
    best = NEVER
    if procedure == "bonferroni":
        level = delta / max(1, len(stats))
        for lam, k, w in stats:
            if binom_pvalue(w, k, alpha) <= level:
                best = max(best, lam)
        return best
    for lam, k, w in stats:  # fixed sequence: most conservative first, stop at first non-rejection
        if binom_pvalue(w, k, alpha) <= delta:
            best = lam
        else:
            break
    return best


def ltt_threshold(scores: list[float], wrong: list[bool], alpha: float, delta: float = 0.1,
                  grid: list[float] | None = None, procedure: str = "fixed_sequence") -> float:
    """Largest grid threshold certified to have false-merge rate <= alpha w.p. >= 1 - delta."""
    grid = sorted(grid or LTT_GRID)
    stats = []
    for lam in grid:
        st = rule_stats(scores, wrong, lam)
        stats.append((lam, st.n_merged, st.n_wrong_merged))
    return _ltt_scan(stats, alpha, delta, procedure)


def ltt_reject_certified(scores: list[float], wrong: list[bool], beta: float, delta: float = 0.1,
                         grid: list[float] | None = None) -> list[float]:
    """Every reject threshold rho certified to have P(correct | s >= rho) <= beta (jointly w.p. >= 1 - delta).

    Patches with s >= rho are rejected (gated patches score 1.0 and are always in the reject set).
    Fixed-sequence from the most conservative rho (1.0: reject only gated patches) downwards; all
    thresholds before the first failure are certified together, so any of them may be used.
    """
    cands = sorted({GATED, *(g for g in (grid or LTT_GRID) if g < GATED)}, reverse=True)
    ok = []
    for rho in cands:
        rej = [b for s, b in zip(scores, wrong) if s >= rho]
        c = sum(1 for b in rej if not b)  # correct patches that would be thrown away
        if binom_pvalue(c, len(rej), beta) <= delta:
            ok.append(rho)
        else:
            break
    return ok


def ltt_reject_threshold(scores: list[float], wrong: list[bool], beta: float, delta: float = 0.1,
                         grid: list[float] | None = None) -> float:
    """Smallest certified reject threshold, or NO_REJECT if none (see ltt_reject_certified)."""
    ok = ltt_reject_certified(scores, wrong, beta, delta, grid)
    return min(ok) if ok else NO_REJECT


@dataclass
class Tiers:
    merge: float    # merge iff s <= merge
    reject: float   # reject iff s >= reject (or gated)

    def tier(self, score: float, gated: bool = False) -> str:
        if gated:
            score = GATED
        if score >= self.reject:  # NO_REJECT (2.0) is above every score: nothing rejected
            return REJECT
        if score <= self.merge and score < GATED:
            return MERGE
        return REVIEW


def tier_thresholds(scores: list[float], wrong: list[bool], alpha: float, beta: float = 0.1,
                    delta: float = 0.1, grid: list[float] | None = None, method: str = "ltt") -> Tiers:
    """Jointly calibrated merge / reject thresholds.

    delta is split in half: with probability >= 1 - delta BOTH hold: the false-merge rate of the
    merge tier is <= alpha and the share of correct patches in the reject tier is <= beta (LTT).
    With method="crc" the merge threshold is the CRC one (expected risk) and only the reject side
    uses delta / 2.
    """
    if method == "crc":
        merge = crc_threshold(scores, wrong, alpha)
    else:
        merge = ltt_threshold(scores, wrong, alpha, delta / 2, grid)
    # Any certified reject threshold is valid; take the smallest one above the merge threshold so the
    # tiers never overlap.
    above = [r for r in ltt_reject_certified(scores, wrong, beta, delta / 2, grid) if r > merge]
    return Tiers(merge=merge, reject=min(above) if above else NO_REJECT)


def threshold(method: str, scores: list[float], wrong: list[bool], alpha: float, delta: float = 0.1,
              grid: list[float] | None = None) -> float:
    if method == "crc":
        return crc_threshold(scores, wrong, alpha)
    if method == "ltt":
        return ltt_threshold(scores, wrong, alpha, delta, grid)
    if method == "ltt_bonferroni":
        return ltt_threshold(scores, wrong, alpha, delta, grid, procedure="bonferroni")
    raise ValueError(f"unknown method {method!r}")


def guarantee_text(method: str, alpha: float, delta: float, n: int) -> str:
    if method == "crc":
        return (f"Conformal risk control on {n} held-out bugs: in expectation, at most {alpha:.0%} of ALL bugs "
                f"end up with a wrong patch auto-merged (a bound on wrong merges per bug, not per merge).")
    return (f"Learn-then-Test on {n} held-out bugs: with probability at least {1 - delta:.0%}, at most "
            f"{alpha:.0%} of auto-merged patches are wrong (false-merge rate), judged by the held-out tests.")
