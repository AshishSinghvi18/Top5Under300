"""Compute composite scores and rank stocks."""

from __future__ import annotations

from typing import Any, Dict, List


def _technical_score(conditions_met: int) -> float:
    return (conditions_met / 6) * 100


def _fundamental_score(fund: Dict[str, Any]) -> float:
    score = 50.0
    if fund.get("roe", 0) > 15:
        score += 15
    if fund.get("pe", float("inf")) < fund.get("sector_median_pe", float("inf")):
        score += 15
    if fund.get("debt_to_equity", float("inf")) < 0.5:
        score += 10
    if fund.get("revenue_growth", 0) > 10:
        score += 10
    return min(score, 100.0)


def _sentiment_score(weighted_count: float, negative_matches: list) -> float:
    score = 50.0 + min(weighted_count * 25, 50)
    score -= 25 * len(negative_matches)
    return max(score, 0.0)


def _sector_score(stock_20d_return: float) -> float:
    score = 50.0 + stock_20d_return * 500
    return max(0.0, min(score, 100.0))


def compute_all_scores(
    fund: Dict[str, Any],
    l1: Any,
    l2: Any,
    l3: Any,
    l4: Any,
    df: Any,
    stock_20d: float,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Return a dict of individual and composite scores (all 0-100)."""
    weights = config.get("scoring", {})

    technical = _technical_score(getattr(l1, "conditions_met", 0))
    fundamental = _fundamental_score(fund)
    sentiment = _sentiment_score(
        getattr(l3, "weighted_count", 0),
        getattr(l3, "negative_matches", []),
    )
    sector = _sector_score(stock_20d)

    composite = (
        technical * weights.get("technical_weight", 0.40)
        + fundamental * weights.get("fundamental_weight", 0.30)
        + sentiment * weights.get("sentiment_weight", 0.20)
        + sector * weights.get("sector_weight", 0.10)
    )

    return {
        "technical_score": round(technical, 2),
        "fundamental_score": round(fundamental, 2),
        "sentiment_score": round(sentiment, 2),
        "sector_score": round(sector, 2),
        "composite_score": round(composite, 2),
    }


def rank_top_n(qualified: List[Any], n: int = 5) -> List[Any]:
    """Sort *qualified* by composite_score descending and return the top *n*."""
    sorted_list = sorted(
        qualified,
        key=lambda item: item.scores.get("composite_score", 0),
        reverse=True,
    )
    return sorted_list[:n]
