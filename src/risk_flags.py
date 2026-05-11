"""Deterministic risk-warning flag generator.

Produces a list of short, human-readable warning strings for a screened stock.
Only flags whose conditions are met are included.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.indicators import compute_rsi
from src.confluence import Layer1Result, Layer2Result, Layer4Result
from src.news_sentiment import SentimentResult
from src.trade_setup import TradeSetup


def generate_risk_flags(
    df: pd.DataFrame,
    fundamentals: Dict[str, Any],
    layer1: Layer1Result,
    layer2: Layer2Result,
    layer3: SentimentResult,
    layer4: Layer4Result,
    setup: TradeSetup,
    promoter_skipped: bool,
    corp_action_skipped: bool,
    asm_check_skipped: bool,
    config: dict,
) -> List[str]:
    """Return a list of risk-warning flag strings that apply to this stock.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame (columns: Open, High, Low, Close, Volume).
    fundamentals : dict
        Keys: pe, debt_to_equity, roe, revenue_growth, market_cap.
    layer1 : Layer1Result
        Technical confluence result.
    layer2 : Layer2Result
        Fundamental check result.
    layer3 : SentimentResult
        News-sentiment result.
    layer4 : Layer4Result
        Sector-momentum result.
    setup : TradeSetup
        Computed trade setup.
    promoter_skipped : bool
        True when promoter-holding data was unavailable.
    corp_action_skipped : bool
        True when corporate-action check was skipped.
    asm_check_skipped : bool
        True when ASM/GSM check was skipped.
    config : dict
        Full application config dict.
    """
    flags: List[str] = []

    tc = config.get("technical", {})
    trade_cfg = config.get("trade", {})
    portfolio_cfg = config.get("portfolio", {})

    # 1. Barely passed technical conditions
    min_conditions = tc.get("min_technical_conditions", 4)
    if layer1.conditions_met == min_conditions:
        flags.append(
            f"\u26a0 Minimum technical conditions ({layer1.conditions_met}/6)"
        )

    # 2. RSI near overbought
    rsi_period = tc.get("rsi_period", 14)
    close = df["Close"]
    rsi_series = compute_rsi(close, rsi_period)
    latest_rsi = rsi_series.iloc[-1] if not rsi_series.empty else np.nan
    if not np.isnan(latest_rsi) and latest_rsi > 65:
        flags.append("\u26a0 RSI near overbought (>65)")

    # 3. High debt-to-equity
    debt_to_equity = fundamentals.get("debt_to_equity")
    if debt_to_equity is not None and debt_to_equity > 1.0:
        flags.append("\u26a0 High debt-to-equity (>1.0)")

    # 4. Low ROE
    roe = fundamentals.get("roe")
    if roe is not None and roe < 0.12:
        flags.append("\u26a0 Low ROE (<12%)")

    # 5. Negative revenue growth
    revenue_growth = fundamentals.get("revenue_growth")
    if revenue_growth is not None and revenue_growth < 0:
        flags.append("\u26a0 Negative revenue growth")

    # 6. PE above sector median
    pe_details = layer2.details
    pe_ok = pe_details.get("pe_ok")
    if pe_ok is False:
        flags.append("\u26a0 PE above sector median")

    # 7. Wide stop loss
    max_sl_pct = trade_cfg.get("max_sl_percent", 0.05)
    if setup.sl_percent > 0.04:
        flags.append(f"\u26a0 Wide stop loss (>{max_sl_pct * 100:.0f}%)")

    # 8. Tight stop loss
    if setup.sl_percent < 0.01:
        flags.append("\u26a0 Tight stop loss (<1%)")

    # 9. High position concentration
    portfolio_size = portfolio_cfg.get("default_size", 100_000)
    if setup.position_value > 0.15 * portfolio_size:
        flags.append("\u26a0 High position concentration (>15%)")

    # 10. Near 52-week high
    high_52w_lookback = trade_cfg.get("high_52w_lookback", 252)
    lookback = min(high_52w_lookback, len(df))
    high_52w = df["High"].iloc[-lookback:].max()
    latest_close = close.iloc[-1]
    if latest_close >= high_52w * 0.95:
        flags.append("\u26a0 Near 52-week high")

    # 11. Low volume trend
    vol_avg_days = tc.get("volume_avg_lookback_days", 20)
    if len(df) >= vol_avg_days:
        vol = df["Volume"]
        avg_vol_20d = vol.iloc[-vol_avg_days:].mean()
        recent_vol = vol.iloc[-5:].mean()
        if avg_vol_20d > 0 and recent_vol < avg_vol_20d:
            flags.append("\u26a0 Low volume trend")

    # 12. Narrow sector outperformance
    sector_diff = layer4.sector_return - layer4.nifty_return
    if 0 < sector_diff <= 0.01:
        flags.append("\u26a0 Narrow sector outperformance")

    # 13. News sentiment neutral
    if layer3.label == "neutral" and layer3.weighted_count < 0.5:
        flags.append("\u26a0 News sentiment neutral (no catalysts)")

    # 14–16. Skipped checks
    if promoter_skipped:
        flags.append("\u26a0 Promoter holding data unavailable")
    if corp_action_skipped:
        flags.append("\u26a0 Corporate action check skipped")
    if asm_check_skipped:
        flags.append("\u26a0 ASM/GSM check skipped")

    # 17. Small market cap
    market_cap = fundamentals.get("market_cap")
    if market_cap is not None and market_cap < 10_000_000_000:
        flags.append("\u26a0 Small market cap (<\u20b91000Cr)")

    return flags
