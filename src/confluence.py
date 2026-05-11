"""4-layer confluence check for the NSE stock screener.

Layer 1 — Technical (6 conditions, configurable minimum to pass)
Layer 2 — Fundamentals (all conditions must pass, skippable when data missing)
Layer 4 — Sector tailwind (sector must outperform NIFTY 50 over 20 days)

Layer 3 (sentiment) lives in src.news_sentiment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from src.indicators import (
    compute_adx,
    compute_macd,
    compute_rsi,
    compute_sma,
    compute_supertrend,
)


@dataclass
class Layer1Result:
    passed: bool
    conditions_met: int
    details: Dict[str, bool] = field(default_factory=dict)


@dataclass
class Layer2Result:
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Layer4Result:
    passed: bool
    sector_return: float
    nifty_return: float
    stock_20d_return: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_20d_return(close_series: pd.Series) -> float:
    """Percentage return over the last 20 trading days.

    Returns (close[-1] / close[-21] - 1) if enough data, else 0.0.
    """
    if len(close_series) < 21:
        return 0.0
    prev = close_series.iloc[-21]
    if prev == 0:
        return 0.0
    return float(close_series.iloc[-1] / prev - 1)


# ---------------------------------------------------------------------------
# Layer 1 — Technical
# ---------------------------------------------------------------------------

def evaluate_layer1_technical(df: pd.DataFrame, config: Dict[str, Any]) -> Layer1Result:
    """Evaluate 6 technical conditions; pass if >= min_technical_conditions met."""
    tc = config["technical"]
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    details: Dict[str, bool] = {}

    # 1. RSI dual-state
    rsi = compute_rsi(close, tc["rsi_period"])
    latest_rsi = rsi.iloc[-1] if not rsi.empty else float("nan")
    state_a = tc["rsi_lower"] <= latest_rsi <= tc["rsi_upper"]
    lookback = tc["rsi_state_b_lookback"]
    recent_rsi = rsi.iloc[-lookback:] if len(rsi) >= lookback else rsi
    state_b = bool(
        (recent_rsi < tc["rsi_lower"]).any()
        and len(rsi) >= 2
        and rsi.iloc[-1] > rsi.iloc[-2]
    )
    details["rsi_dual_state"] = state_a or state_b

    # 2. MACD bullish crossover within lookback
    macd_line, signal_line, _ = compute_macd(
        close, tc["macd_fast"], tc["macd_slow"], tc["macd_signal"]
    )
    cross_lookback = tc["macd_crossover_lookback"]
    macd_cross = False
    if len(macd_line) > cross_lookback:
        for i in range(-cross_lookback, 0):
            if macd_line.iloc[i] > signal_line.iloc[i] and macd_line.iloc[i - 1] <= signal_line.iloc[i - 1]:
                macd_cross = True
                break
    details["macd_crossover"] = macd_cross

    # 3. SMA stack: close > SMA(short) > SMA(long), close > SMA(trend)
    sma_short = compute_sma(close, tc["sma_short"])
    sma_long = compute_sma(close, tc["sma_long"])
    sma_trend = compute_sma(close, tc["sma_trend"])
    latest_close = close.iloc[-1]
    details["sma_stack"] = bool(
        not pd.isna(sma_short.iloc[-1])
        and not pd.isna(sma_long.iloc[-1])
        and not pd.isna(sma_trend.iloc[-1])
        and latest_close > sma_short.iloc[-1] > sma_long.iloc[-1]
        and latest_close > sma_trend.iloc[-1]
    )

    # 4. Volume surge in recent days
    vol_lookback = tc["volume_surge_lookback_days"]
    vol_avg_days = tc["volume_avg_lookback_days"]
    vol_mult = tc["volume_surge_multiplier"]
    if len(volume) > vol_avg_days:
        avg_vol = volume.iloc[-(vol_avg_days + vol_lookback):-vol_lookback].mean()
        recent_vol = volume.iloc[-vol_lookback:]
        details["volume_surge"] = bool((recent_vol > vol_mult * avg_vol).any())
    else:
        details["volume_surge"] = False

    # 5. Supertrend bullish for min_sessions consecutive sessions
    st = compute_supertrend(df, tc["supertrend_period"], tc["supertrend_multiplier"])
    min_sessions = tc["supertrend_min_sessions"]
    direction = st["direction"]
    if len(direction) >= min_sessions:
        tail = direction.iloc[-min_sessions:]
        details["supertrend_bullish"] = bool((tail == 1).all())
    else:
        details["supertrend_bullish"] = False

    # 6. ADX > threshold
    adx = compute_adx(high, low, close, tc["adx_period"])
    details["adx_strong"] = bool(
        not pd.isna(adx.iloc[-1]) and adx.iloc[-1] > tc["adx_threshold"]
    )

    conditions_met = sum(details.values())
    passed = conditions_met >= tc["min_technical_conditions"]
    return Layer1Result(passed=passed, conditions_met=conditions_met, details=details)


# ---------------------------------------------------------------------------
# Layer 2 — Fundamentals
# ---------------------------------------------------------------------------

def evaluate_layer2_fundamentals(
    fund: Dict[str, Any],
    promoter: Optional[Dict[str, Any]],
    sector_median_pe: float,
    config: Dict[str, Any],
) -> Layer2Result:
    """All fundamental checks must pass (skip when data is None)."""
    fc = config["fundamental"]
    details: Dict[str, Any] = {}
    checks_passed = True

    # 1. PE < sector_median_pe * max_pe_multiplier
    pe = fund.get("pe")
    if pe is not None:
        threshold = sector_median_pe * fc["max_pe_multiplier"]
        ok = pe < threshold
        details["pe"] = {"value": pe, "threshold": threshold, "passed": ok}
        if not ok:
            checks_passed = False
    else:
        details["pe"] = {"value": None, "skipped": True}

    # 2. Debt-to-equity < max_debt_to_equity
    de = fund.get("debt_to_equity")
    if de is not None:
        ok = de < fc["max_debt_to_equity"]
        details["debt_to_equity"] = {"value": de, "threshold": fc["max_debt_to_equity"], "passed": ok}
        if not ok:
            checks_passed = False
    else:
        details["debt_to_equity"] = {"value": None, "skipped": True}

    # 3. ROE > min_roe
    roe = fund.get("roe")
    if roe is not None:
        ok = roe > fc["min_roe"]
        details["roe"] = {"value": roe, "threshold": fc["min_roe"], "passed": ok}
        if not ok:
            checks_passed = False
    else:
        details["roe"] = {"value": None, "skipped": True}

    # 4. Revenue growth > min_revenue_growth
    rev_growth = fund.get("revenue_growth")
    if rev_growth is not None:
        ok = rev_growth > fc["min_revenue_growth"]
        details["revenue_growth"] = {"value": rev_growth, "threshold": fc["min_revenue_growth"], "passed": ok}
        if not ok:
            checks_passed = False
    else:
        details["revenue_growth"] = {"value": None, "skipped": True}

    # 5. Promoter holding not dropping significantly
    if promoter is not None:
        current = promoter.get("current")
        previous = promoter.get("previous")
        if current is not None and previous is not None:
            tolerance = fc["promoter_holding_tolerance_pct"]
            ok = current >= previous + tolerance
            details["promoter_holding"] = {
                "current": current,
                "previous": previous,
                "tolerance": tolerance,
                "passed": ok,
            }
            if not ok:
                checks_passed = False
        else:
            details["promoter_holding"] = {"skipped": True}
    else:
        details["promoter_holding"] = {"skipped": True}

    return Layer2Result(passed=checks_passed, details=details)


# ---------------------------------------------------------------------------
# Layer 4 — Sector tailwind
# ---------------------------------------------------------------------------

def evaluate_layer4_sector(
    stock_20d: float,
    sector_idx: str,
    sector_returns: Dict[str, float],
    *,
    nifty_default: str,
) -> Layer4Result:
    """Sector index must outperform NIFTY 50, and stock must have positive 20d return."""
    sector_ret = sector_returns.get(sector_idx, float("nan"))
    nifty_ret = sector_returns.get(nifty_default, float("nan"))

    import math

    if math.isnan(sector_ret) or math.isnan(nifty_ret):
        passed = False
    else:
        passed = sector_ret >= nifty_ret and stock_20d > 0

    return Layer4Result(
        passed=passed,
        sector_return=sector_ret,
        nifty_return=nifty_ret,
        stock_20d_return=stock_20d,
    )
