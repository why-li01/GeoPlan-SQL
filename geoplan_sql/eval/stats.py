"""Wilson confidence interval and McNemar test helpers."""

from __future__ import annotations

from typing import Tuple

from scipy.stats import norm


def wilson_interval(
    successes: int,
    n: int,
    *,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Return (point_estimate, lower, upper) for a binomial proportion."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    z = float(norm.ppf(1 - (1 - confidence) / 2))
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z / denom) * ((phat * (1 - phat) / n + z**2 / (4 * n**2)) ** 0.5)
    return phat, max(0.0, center - margin), min(1.0, center + margin)


def mcnemar_exact(
    both_correct: int,
    only_a: int,
    only_b: int,
    both_wrong: int,
) -> dict:
    """McNemar exact test on discordant pairs (continuity-corrected χ² optional).

    Returns counts and two-sided p-value from binomial on discordants.
    """
    from scipy.stats import binomtest

    n_disc = only_a + only_b
    if n_disc == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(only_a, n_disc, p=0.5, alternative="two-sided").pvalue)

    return {
        "both_correct": both_correct,
        "only_a": only_a,
        "only_b": only_b,
        "both_wrong": both_wrong,
        "n_discordant": n_disc,
        "p_value": p_value,
    }
