"""Trade setup computation: entry, stop loss, targets, and position sizing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.indicators import compute_atr, compute_sma


@dataclass
class TradeSetup:
    """Computed trade setup for a stock."""

    entry: float
    stop_loss: float
    sl_percent: float
    target1: float
    target2: float
    rr1: float
    rr2: float
    position_size: int
    position_value: float
    risk_amount: float
    validity_days: int
    atr: float


def compute_trade_setup(
    df: pd.DataFrame, config: dict, portfolio_size: float
) -> Optional[TradeSetup]:
    """Compute a trade setup from OHLCV data and config.

    Returns None if the setup is invalid (entry <= stop_loss,
    position_size <= 0, or sl_percent is NaN).
    """
    trade_cfg = config.get("trade", {})

    max_sl_pct = trade_cfg.get("max_sl_percent", 0.05)
    min_sl_pct = trade_cfg.get("min_sl_percent", 0.005)
    target1_rr = trade_cfg.get("target1_rr", 1.5)
    target2_rr = trade_cfg.get("target2_rr", 2.5)
    entry_sma_tol = trade_cfg.get("entry_sma_tolerance", 0.02)
    atr_period = trade_cfg.get("atr_period", 14)
    atr_mult = trade_cfg.get("atr_multiplier", 2.0)
    swing_low_lb = trade_cfg.get("swing_low_lookback", 10)
    max_pos_pct = trade_cfg.get("max_position_percent", 0.20)
    risk_per_trade = trade_cfg.get("risk_per_trade", 0.02)
    validity_days = trade_cfg.get("validity_trading_days", 2)

    if df is None or df.empty or len(df) < max(atr_period, swing_low_lb):
        return None

    # --- Entry ---
    close = df["Close"]
    latest_close = float(close.iloc[-1])

    sma20 = compute_sma(close, 20)
    latest_sma20 = float(sma20.iloc[-1]) if not np.isnan(sma20.iloc[-1]) else None

    if latest_sma20 is not None and abs(latest_close - latest_sma20) / latest_close <= entry_sma_tol:
        entry = latest_sma20
    else:
        entry = latest_close

    # --- ATR ---
    atr_series = compute_atr(df["High"], df["Low"], close, atr_period)
    current_atr = float(atr_series.iloc[-1])
    if np.isnan(current_atr) or current_atr <= 0:
        return None

    # --- Stop Loss ---
    swing_low = float(df["Low"].iloc[-swing_low_lb:].min())
    atr_stop = entry - current_atr * atr_mult
    raw_sl = max(swing_low, atr_stop)

    sl_pct = (entry - raw_sl) / entry
    if np.isnan(sl_pct):
        return None

    sl_pct = max(min_sl_pct, min(sl_pct, max_sl_pct))
    stop_loss = entry * (1 - sl_pct)

    # --- Validate entry > stop_loss ---
    if entry <= stop_loss:
        return None

    risk_per_share = entry - stop_loss

    # --- Targets ---
    target1 = entry + risk_per_share * target1_rr
    target2 = entry + risk_per_share * target2_rr

    # --- Risk-reward ratios ---
    rr1 = target1_rr
    rr2 = target2_rr

    # --- Position sizing ---
    risk_amount = portfolio_size * risk_per_trade
    position_size = math.floor(risk_amount / risk_per_share)

    max_shares = math.floor(max_pos_pct * portfolio_size / entry)
    position_size = min(position_size, max_shares)

    if position_size <= 0:
        return None

    position_value = position_size * entry

    return TradeSetup(
        entry=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        sl_percent=round(sl_pct, 4),
        target1=round(target1, 2),
        target2=round(target2, 2),
        rr1=round(rr1, 2),
        rr2=round(rr2, 2),
        position_size=position_size,
        position_value=round(position_value, 2),
        risk_amount=round(risk_amount, 2),
        validity_days=validity_days,
        atr=round(current_atr, 2),
    )
