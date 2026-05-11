from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    passed: bool
    rejection_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


def apply_hard_filters(
    *,
    symbol: str,
    df: pd.DataFrame,
    fundamentals: Dict[str, Any],
    series_code: str,
    asm_gsm_set: Optional[Set[str]],
    corp_actions: Optional[List[Dict[str, Any]]],
    config: Dict[str, Any],
) -> FilterResult:
    """Run all 8 hard filters and return a FilterResult."""

    cfg = config["screener"]
    min_price: float = cfg["min_price"]
    max_price: float = cfg["max_price"]
    min_avg_volume_20d: int = cfg["min_avg_volume_20d"]
    min_market_cap: float = cfg["min_market_cap"]
    min_history_days: int = cfg["min_history_days"]
    allowed_series: List[str] = cfg["allowed_series"]

    warnings: List[str] = []

    # --- Filter 1: Price range ---
    if df.empty:
        return FilterResult(passed=False, rejection_reason=f"{symbol}: no price data available")

    latest_close = df["Close"].dropna().iloc[-1] if not df["Close"].dropna().empty else None
    if latest_close is None:
        return FilterResult(passed=False, rejection_reason=f"{symbol}: no valid close price")

    if latest_close < min_price or latest_close > max_price:
        return FilterResult(
            passed=False,
            rejection_reason=(
                f"{symbol}: price {latest_close:.2f} outside "
                f"[{min_price}, {max_price}]"
            ),
        )

    if latest_close >= max_price * 0.95:
        warnings.append(
            f"{symbol}: price {latest_close:.2f} within 5% of max {max_price}"
        )

    # --- Filter 2: Series ---
    if series_code not in allowed_series:
        return FilterResult(
            passed=False,
            rejection_reason=f"{symbol}: series '{series_code}' not in {allowed_series}",
        )

    # --- Filter 3: History length ---
    if len(df) < min_history_days:
        return FilterResult(
            passed=False,
            rejection_reason=(
                f"{symbol}: only {len(df)} days of history, "
                f"need {min_history_days}"
            ),
        )

    # --- Filter 4: Average volume (20-day) ---
    volume_series = df["Volume"].tail(20)
    avg_volume = volume_series.mean()

    if pd.isna(avg_volume) or avg_volume < min_avg_volume_20d:
        return FilterResult(
            passed=False,
            rejection_reason=(
                f"{symbol}: 20-day avg volume {avg_volume:.0f} "
                f"< {min_avg_volume_20d}"
            ),
            warnings=warnings,
        )

    if avg_volume < min_avg_volume_20d * 1.2:
        warnings.append(
            f"{symbol}: 20-day avg volume {avg_volume:.0f} within 20% of "
            f"threshold {min_avg_volume_20d}"
        )

    # --- Filter 5: Market cap ---
    market_cap = fundamentals.get("market_cap")
    if market_cap is not None:
        if market_cap < min_market_cap:
            return FilterResult(
                passed=False,
                rejection_reason=(
                    f"{symbol}: market cap {market_cap:.0f} "
                    f"< {min_market_cap:.0f}"
                ),
                warnings=warnings,
            )
        if market_cap < min_market_cap * 1.2:
            warnings.append(
                f"{symbol}: market cap {market_cap:.0f} within 20% of "
                f"threshold {min_market_cap:.0f}"
            )

    # --- Filter 6: ASM/GSM list ---
    if asm_gsm_set is not None and symbol in asm_gsm_set:
        return FilterResult(
            passed=False,
            rejection_reason=f"{symbol}: in ASM/GSM surveillance list",
            warnings=warnings,
        )

    # --- Filter 7: Corporate actions ---
    if corp_actions is not None and len(corp_actions) > 0:
        return FilterResult(
            passed=False,
            rejection_reason=(
                f"{symbol}: upcoming corporate action within lookahead window"
            ),
            warnings=warnings,
        )

    # --- Filter 8: Price data quality ---
    nan_ratio = df["Close"].isna().mean()
    if nan_ratio > 0.10:
        return FilterResult(
            passed=False,
            rejection_reason=(
                f"{symbol}: {nan_ratio:.1%} NaN in Close column "
                f"(threshold 10%)"
            ),
            warnings=warnings,
        )

    return FilterResult(passed=True, warnings=warnings)
