from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo  # FIX #2: IST-aware incomplete-bar handling

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import ta
import yaml
import yfinance as yf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

# FIX #2: NSE regular session closes 15:30 IST. A daily bar dated "today" fetched
# before this time is a partial/incomplete candle and must not drive signals.
IST = ZoneInfo("Asia/Kolkata")
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

SECTOR_PE_FALLBACK = {
    "Financial Services": 15,
    "Technology": 28,
    "Healthcare": 30,
    "Consumer Cyclical": 25,
    "Consumer Defensive": 40,
    "Energy": 15,
    "Basic Materials": 12,
    "Industrials": 22,
    "Utilities": 18,
    "Communication Services": 20,
    "Real Estate": 30,
    "Default": 22,
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "screener": {
        "min_price": 50.0,
        "max_price": 300.0,
        "min_avg_volume_20d": 100000,
        "min_history_days": 200,
    },
    "technical": {
        "rsi_lower": 50,
        "rsi_upper": 70,
        "min_technical_conditions": 3,
        "volume_surge_multiplier": 1.5,
        "adx_threshold": 25,
        "macd_crossover_lookback": 5,
    },
    "fundamental": {
        "max_pe_multiplier": 1.2,
        "max_debt_to_equity": 1.5,
        "min_roe": 0.10,
        "min_revenue_growth": 0.0,
        # FIX #4 (optional knob): minimum fundamental checks that must pass.
        # auto_pass_when_thin preserves the original behaviour: if Yahoo returns
        # fewer than min_fundamental_pass usable fields, the stock passes on a
        # risk-flag rather than being dropped. Set auto_pass_when_thin=false to
        # require real fundamental confirmation instead.
        "min_fundamental_pass": 2,
        "auto_pass_when_thin": True,
    },
    "trade": {
        "atr_multiplier": 2.0,
        "target1_rr": 1.5,
        "target2_rr": 2.5,
        "target3_rr": 3.5,
        "risk_per_trade": 0.02,
        "validity_trading_days": 2,
        "max_position_percent": 0.20,
        "min_sl_percent": 0.005,
        "max_sl_percent": 0.05,
        # FIX #3: the 3-10% band is now an actual SELECTION filter, not a display
        # clamp. A stock qualifies only if its expected N-session move (ATR-based)
        # lands in [return_band_min, return_band_max]. Set enforce_return_band=false
        # to revert to pure-momentum selection.
        "return_band_min": 3.0,
        "return_band_max": 10.0,
        "enforce_return_band": True,
        "expected_move_sessions": 2,
    },
    "scoring": {
        "max_buy_signals": 5,
        "technical_weight": 0.40,
        "fundamental_weight": 0.30,
    },
    "portfolio": {
        "default_size": 100000,
    },
}

LOGGER = logging.getLogger("nse_scanner")
CONSOLE = Console()


@dataclass
class StockResult:
    symbol: str
    company: str
    sector: str
    current_price: float
    technical_score: float
    fundamental_score: float
    setup_score: float
    confidence_score: float
    entry: float
    stop_loss: float
    stop_loss_pct: float
    target1: float
    target2: float
    target3: float
    rr1: float
    rr2: float
    rr3: float
    return_min: float
    return_max: float
    expected_move_pct: float  # FIX #3: ATR-based N-session expected move used for band gate
    position_qty: int
    position_value: float
    validity_days: int
    avg_volume_20d: float
    atr: float
    atr_pct: float
    rsi: float
    adx: float
    five_day_return: float
    roc10: float
    bb_position: float
    technical_pass_count: int
    technical_condition_count: int
    fundamental_pass_count: int
    fundamental_condition_count: int
    trigger_events: List[str] = field(default_factory=list)
    bullish_patterns: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    technical_conditions: Dict[str, bool] = field(default_factory=dict)
    fundamental_conditions: Dict[str, Optional[bool]] = field(default_factory=dict)
    raw_fundamentals: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_csv_record(self) -> Dict[str, Any]:
        return {
            "Symbol": self.symbol,
            "Company": self.company,
            "Price": round(self.current_price, 2),
            "Sector": self.sector,
            "TechnicalScore": round(self.technical_score, 2),
            "FundamentalScore": round(self.fundamental_score, 2),
            "ConfidenceScore": round(self.confidence_score, 2),
            "Entry": round(self.entry, 2),
            "StopLoss": round(self.stop_loss, 2),
            "Target1": round(self.target1, 2),
            "Target2": round(self.target2, 2),
            "Target3": round(self.target3, 2),
            "ExpectedMovePct": round(self.expected_move_pct, 2),  # FIX #3
            "ReturnToT1Pct": round(self.return_min, 2),
            "ReturnToT3Pct": round(self.return_max, 2),
            "TriggerEvents": "; ".join(self.trigger_events),
            "RiskFlags": "; ".join(self.risk_flags),
        }

    def to_json_record(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, float):
                payload[key] = None if math.isnan(value) else round(value, 4)
        return payload


class ScanError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSE under-₹300 stock scanner")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--portfolio", type=float, default=None, help="Portfolio size in INR")
    parser.add_argument("--max-price", type=float, default=None, help="Override maximum stock price")
    parser.add_argument("--min-price", type=float, default=None, help="Override minimum stock price")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--limit", type=int, default=None, help="Scan only the first N stocks")
    parser.add_argument(
        "--no-return-band",
        action="store_true",
        help="Disable the 3-10%% expected-move filter (revert to pure momentum selection)",
    )
    parser.add_argument(
        "--keep-incomplete-bar",
        action="store_true",
        help="Do NOT drop today's partial bar (use only if you run against completed data)",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("curl_cffi").setLevel(logging.CRITICAL)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str, args: argparse.Namespace) -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    path = Path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        config = deep_merge(config, loaded)
    config["sector_pe_fallback"] = deep_merge(SECTOR_PE_FALLBACK, config.get("sector_pe_fallback", {}))

    if args.min_price is not None:
        config["screener"]["min_price"] = float(args.min_price)
    if args.max_price is not None:
        config["screener"]["max_price"] = float(args.max_price)
    if args.portfolio is not None:
        config.setdefault("portfolio", {})["default_size"] = float(args.portfolio)
    if getattr(args, "no_return_band", False):
        config["trade"]["enforce_return_band"] = False
    config["screener"]["max_price"] = min(300.0, float(config["screener"]["max_price"]))
    return config


def print_disclaimer() -> None:
    text = Text()
    text.append("NSE Under ₹300 Momentum Scanner\n", style="bold yellow")
    text.append(
        "Educational output only. Uses Yahoo Finance data, which may be delayed, incomplete, or unavailable. "
        "Always verify liquidity, corporate actions, and risk before trading.",
        style="white",
    )
    CONSOLE.print(Panel(text, border_style="yellow", box=box.ROUNDED))


def load_universe(csv_path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Universe file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"Symbol", "Series", "ISIN", "Sector"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    df = df.dropna(subset=["Symbol"]).copy()
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper().str.replace(".NS", "", regex=False)
    df["Series"] = df["Series"].fillna("EQ").astype(str).str.strip().str.upper()
    df["Sector"] = df["Sector"].fillna("Default").astype(str).str.strip()
    df = df[df["Symbol"] != ""]
    df = df.drop_duplicates(subset=["Symbol"]).reset_index(drop=True)
    if limit:
        df = df.head(limit)
    return df


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pct(value: float) -> str:
    return f"{value:.2f}%"


def rupees(value: float) -> str:
    return f"₹{value:,.2f}"


def drop_incomplete_bar(history: pd.DataFrame) -> pd.DataFrame:
    """FIX #2: Drop today's partial daily bar if the session has not yet closed.

    Yahoo returns a live, still-forming candle for the current day while the
    market is open. Every signal in this scanner reads df.iloc[-1], so leaving
    that partial bar in place corrupts RSI, the MACD cross, the volume surge,
    ATR and every candle pattern. This is the daily-timeframe analogue of the
    intraday partial-bar problem. On weekends/holidays the last bar's date will
    not equal today, so nothing is dropped.
    """
    if history is None or history.empty:
        return history
    now_ist = datetime.now(IST)
    last_ts = pd.Timestamp(history.index[-1])
    if last_ts.tzinfo is not None:
        last_date = last_ts.tz_convert(IST).date()
    else:
        last_date = last_ts.date()
    close_dt = now_ist.replace(
        hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0
    )
    if last_date == now_ist.date() and now_ist < close_dt:
        return history.iloc[:-1].copy()
    return history


def fetch_stock_data(
    symbol: str, drop_partial: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ticker = yf.Ticker(f"{symbol}.NS")
    try:
        # FIX #5 (price correctness): auto_adjust=True back-adjusts historical
        # OHLC for splits/bonuses/dividends. With the old auto_adjust=False, any
        # corporate action in the trailing year injects a false gap into the raw
        # series, corrupting RSI/MACD/SMA/ATR for weeks around the ex-date. The
        # MOST RECENT bar's adjustment factor is ~1.0, so the displayed current
        # price stays the true last traded price; only history is rescaled.
        history = ticker.history(period="1y", auto_adjust=True)
    except Exception as exc:
        raise ScanError(f"Yahoo Finance history fetch failed: {exc}") from exc
    if history is None or history.empty:
        raise ScanError("No price history returned by yfinance")

    history = history.rename(columns=str.title).copy()
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in required_cols if col not in history.columns]
    if missing_cols:
        raise ScanError(f"History missing required columns: {missing_cols}")

    history = history[required_cols].dropna().copy()
    history = history[history["Volume"] >= 0]

    if drop_partial:  # FIX #2
        history = drop_incomplete_bar(history)

    if len(history) < 60:
        raise ScanError("Insufficient history for indicator calculations")

    try:
        info = ticker.info or {}
    except Exception as exc:
        LOGGER.debug("info fetch failed for %s: %s", symbol, exc)
        info = {}

    return history, info


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["RSI14"] = ta.momentum.RSIIndicator(close=enriched["Close"], window=14).rsi()

    macd = ta.trend.MACD(close=enriched["Close"], window_fast=12, window_slow=26, window_sign=9)
    enriched["MACD"] = macd.macd()
    enriched["MACD_SIGNAL"] = macd.macd_signal()
    enriched["MACD_HIST"] = macd.macd_diff()

    enriched["SMA20"] = enriched["Close"].rolling(20).mean()
    enriched["SMA50"] = enriched["Close"].rolling(50).mean()
    enriched["VOL20"] = enriched["Volume"].rolling(20).mean()

    enriched["ADX14"] = ta.trend.ADXIndicator(
        high=enriched["High"],
        low=enriched["Low"],
        close=enriched["Close"],
        window=14,
    ).adx()

    bb = ta.volatility.BollingerBands(close=enriched["Close"], window=20, window_dev=2)
    enriched["BB_HIGH"] = bb.bollinger_hband()
    enriched["BB_LOW"] = bb.bollinger_lband()
    enriched["BB_MID"] = bb.bollinger_mavg()

    enriched["ATR14"] = ta.volatility.AverageTrueRange(
        high=enriched["High"],
        low=enriched["Low"],
        close=enriched["Close"],
        window=14,
    ).average_true_range()

    enriched["ROC10"] = ta.momentum.ROCIndicator(close=enriched["Close"], window=10).roc()
    enriched["RET5"] = enriched["Close"].pct_change(5) * 100
    enriched["RET10"] = enriched["Close"].pct_change(10) * 100
    enriched["RANGE10_PCT"] = (
        enriched["High"].rolling(10).max() - enriched["Low"].rolling(10).min()
    ) / enriched["Close"] * 100
    return enriched


def validate_indicator_snapshot(df: pd.DataFrame) -> None:
    latest = df.iloc[-1]
    required = [
        "RSI14",
        "MACD",
        "MACD_SIGNAL",
        "SMA20",
        "SMA50",
        "VOL20",
        "ADX14",
        "BB_HIGH",
        "BB_LOW",
        "ATR14",
        "ROC10",
        "RET5",
    ]
    missing = [col for col in required if pd.isna(latest[col])]
    if missing:
        raise ScanError(f"Indicator snapshot contains NaN values: {missing}")


def detect_recent_macd_crossover(df: pd.DataFrame, lookback: int) -> bool:
    macd = df["MACD"]
    signal = df["MACD_SIGNAL"]
    if len(df) < lookback + 2:
        return False

    recent = df.iloc[-(lookback + 1) :].copy()
    crossover = (recent["MACD"] > recent["MACD_SIGNAL"]) & (
        recent["MACD"].shift(1) <= recent["MACD_SIGNAL"].shift(1)
    )
    return bool(crossover.iloc[1:].any())


def evaluate_layer1(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    latest = df.iloc[-1]
    price = float(latest["Close"])
    avg_volume_20d = float(df["Volume"].rolling(20).mean().iloc[-1])
    passed = (
        config["screener"]["min_price"] <= price <= config["screener"]["max_price"]
        and avg_volume_20d > float(config["screener"]["min_avg_volume_20d"])
    )
    return {
        "passed": passed,
        "price": price,
        "avg_volume_20d": avg_volume_20d,
    }


def evaluate_layer2(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    latest = df.iloc[-1]
    rsi_lower = float(config["technical"]["rsi_lower"])
    rsi_upper = float(config["technical"]["rsi_upper"])
    lookback = int(config["technical"]["macd_crossover_lookback"])
    volume_multiplier = float(config["technical"]["volume_surge_multiplier"])
    adx_threshold = float(config["technical"]["adx_threshold"])

    conditions = {
        "RSI 50-70 bullish zone": rsi_lower <= float(latest["RSI14"]) <= rsi_upper,
        "MACD bullish crossover within 5 days": bool(
            latest["MACD"] > latest["MACD_SIGNAL"] and detect_recent_macd_crossover(df, lookback)
        ),
        "Price above SMA20 and SMA20 above SMA50": bool(
            latest["Close"] > latest["SMA20"] > latest["SMA50"]
        ),
        "Volume surge above 1.5x 20D average": bool(
            latest["Volume"] > volume_multiplier * latest["VOL20"]
        ),
        "ADX above 25": bool(latest["ADX14"] > adx_threshold),
    }

    pass_count = sum(bool(value) for value in conditions.values())
    triggers = [name for name, passed in conditions.items() if passed]
    score = round(pass_count / max(len(conditions), 1) * 100, 2)
    return {
        "passed": pass_count >= int(config["technical"]["min_technical_conditions"]),
        "conditions": conditions,
        "pass_count": pass_count,
        "score": score,
        "triggers": triggers,
    }


def normalize_debt_to_equity(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    normalized = float(value)
    if normalized > 10:
        normalized = normalized / 100.0
    return normalized


def normalize_pct_decimal(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    normalized = float(value)
    if abs(normalized) > 1.0:
        normalized = normalized / 100.0
    return normalized


def score_boolean_condition(value: Optional[bool]) -> float:
    if value is None:
        return 50.0
    return 100.0 if value else 0.0


def evaluate_layer3(info: Dict[str, Any], fallback_sector: str, config: Dict[str, Any]) -> Dict[str, Any]:
    sector = str(info.get("sector") or fallback_sector or "Default").strip() or "Default"
    sector_median = float(config["sector_pe_fallback"].get(sector, config["sector_pe_fallback"]["Default"]))

    pe = safe_float(info.get("trailingPE"))
    debt_to_equity = normalize_debt_to_equity(safe_float(info.get("debtToEquity")))
    roe = normalize_pct_decimal(safe_float(info.get("returnOnEquity")))
    revenue_growth = normalize_pct_decimal(safe_float(info.get("revenueGrowth")))

    conditions: Dict[str, Optional[bool]] = {
        f"P/E below {config['fundamental']['max_pe_multiplier']:.1f}x sector median": None,
        "Debt-to-equity below 1.5": None,
        "ROE above 10%": None,
        "Positive revenue growth": None,
    }

    if pe is not None and pe > 0:
        conditions[f"P/E below {config['fundamental']['max_pe_multiplier']:.1f}x sector median"] = (
            pe < sector_median * float(config["fundamental"]["max_pe_multiplier"])
        )
    if debt_to_equity is not None:
        conditions["Debt-to-equity below 1.5"] = debt_to_equity < float(config["fundamental"]["max_debt_to_equity"])
    if roe is not None:
        conditions["ROE above 10%"] = roe > float(config["fundamental"]["min_roe"])
    if revenue_growth is not None:
        conditions["Positive revenue growth"] = revenue_growth > float(config["fundamental"].get("min_revenue_growth", 0.0))

    available_checks = sum(value is not None for value in conditions.values())
    pass_count = sum(value is True for value in conditions.values())
    score = round(sum(score_boolean_condition(value) for value in conditions.values()) / len(conditions), 2)

    # FIX #4 (optional knob): original behaviour was `pass_count >= 2 or available_checks < 2`,
    # i.e. thin-data stocks auto-passed. That is preserved by default but now controllable.
    min_pass = int(config["fundamental"].get("min_fundamental_pass", 2))
    auto_pass_thin = bool(config["fundamental"].get("auto_pass_when_thin", True))
    passed = pass_count >= min_pass or (auto_pass_thin and available_checks < min_pass)

    return {
        "passed": passed,
        "sector": sector,
        "sector_median_pe": sector_median,
        "pass_count": pass_count,
        "available_checks": available_checks,
        "score": score,
        "conditions": conditions,
        "raw": {
            "trailing_pe": pe,
            "debt_to_equity": debt_to_equity,
            "roe_pct": None if roe is None else roe * 100,
            "revenue_growth_pct": None if revenue_growth is None else revenue_growth * 100,
        },
    }


def score_atr_band(atr_pct: float) -> float:
    if 2.0 <= atr_pct <= 5.0:
        return 100.0
    if 1.5 <= atr_pct < 2.0 or 5.0 < atr_pct <= 6.0:
        return 75.0
    if 1.0 <= atr_pct < 1.5 or 6.0 < atr_pct <= 7.5:
        return 55.0
    return 25.0


def evaluate_layer4(df: pd.DataFrame, layer2: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    latest = df.iloc[-1]
    close = float(latest["Close"])
    atr = float(latest["ATR14"])
    atr_pct = (atr / close) * 100 if close else 0.0
    five_day_return = float(latest["RET5"])
    roc10 = float(latest["ROC10"])
    bb_high = float(latest["BB_HIGH"])
    bb_low = float(latest["BB_LOW"])
    bb_position = 0.5
    if bb_high > bb_low:
        bb_position = (close - bb_low) / (bb_high - bb_low)

    # FIX #3: realistic expected move over the trade's validity horizon.
    # sigma_N ≈ ATR * sqrt(N). This is what the 3-10% band now actually screens on.
    sessions = int(
        config["trade"].get(
            "expected_move_sessions", config["trade"].get("validity_trading_days", 2)
        )
    )
    expected_move_pct = (atr / close) * math.sqrt(max(1, sessions)) * 100 if close else 0.0
    band_min = float(config["trade"].get("return_band_min", 3.0))
    band_max = float(config["trade"].get("return_band_max", 10.0))
    in_band = band_min <= expected_move_pct <= band_max

    atr_score = score_atr_band(atr_pct)
    return_score = clamp(50 + five_day_return * 8, 0, 100)
    roc_score = clamp(50 + roc10 * 6, 0, 100)

    if bb_position <= 0.35:
        bb_score = 85.0
        bb_note = "Near lower Bollinger Band bounce zone"
    elif bb_position >= 0.80 and layer2["conditions"].get("Volume surge above 1.5x 20D average", False):
        bb_score = 92.0
        bb_note = "Upper Bollinger breakout with volume"
    elif bb_position >= 0.80:
        bb_score = 70.0
        bb_note = "Near upper Bollinger Band"
    else:
        bb_score = 60.0
        bb_note = "Mid-band continuation structure"

    setup_score = round(np.mean([atr_score, return_score, roc_score, bb_score]), 2)
    triggers = []
    if five_day_return > 0:
        triggers.append(f"5-day momentum positive ({five_day_return:.2f}%)")
    if roc10 > 0:
        triggers.append(f"10-day ROC positive ({roc10:.2f}%)")
    if 2.0 <= atr_pct <= 5.0:
        triggers.append(f"ATR in sweet spot ({atr_pct:.2f}% of price)")
    triggers.append(f"Expected {sessions}-session move ~{expected_move_pct:.2f}%")
    triggers.append(bb_note)

    return {
        "score": setup_score,
        "atr": atr,
        "atr_pct": atr_pct,
        "expected_move_pct": expected_move_pct,  # FIX #3
        "in_band": in_band,  # FIX #3
        "band_min": band_min,
        "band_max": band_max,
        "sessions": sessions,
        "five_day_return": five_day_return,
        "roc10": roc10,
        "bb_position": bb_position,
        "triggers": triggers,
    }


def candle_parts(candle: pd.Series) -> Dict[str, float]:
    open_price = float(candle["Open"])
    close_price = float(candle["Close"])
    high_price = float(candle["High"])
    low_price = float(candle["Low"])
    body = abs(close_price - open_price)
    total_range = max(high_price - low_price, 1e-9)
    upper_shadow = high_price - max(open_price, close_price)
    lower_shadow = min(open_price, close_price) - low_price
    return {
        "body": body,
        "range": total_range,
        "upper": upper_shadow,
        "lower": lower_shadow,
        "open": open_price,
        "close": close_price,
        "high": high_price,
        "low": low_price,
    }


def detect_hammer(df: pd.DataFrame) -> bool:
    candle = candle_parts(df.iloc[-1])
    return (
        candle["body"] < 0.30 * candle["range"]
        and candle["lower"] > 2 * candle["body"]
        and candle["upper"] <= 0.20 * candle["range"]
    )


def detect_bullish_engulfing(df: pd.DataFrame) -> bool:
    prev = candle_parts(df.iloc[-2])
    curr = candle_parts(df.iloc[-1])
    return (
        prev["close"] < prev["open"]
        and curr["close"] > curr["open"]
        and curr["open"] <= prev["close"]
        and curr["close"] >= prev["open"]
    )


def detect_morning_star(df: pd.DataFrame) -> bool:
    c1 = candle_parts(df.iloc[-3])
    c2 = candle_parts(df.iloc[-2])
    c3 = candle_parts(df.iloc[-1])
    midpoint_c1 = (c1["open"] + c1["close"]) / 2
    return (
        c1["close"] < c1["open"]
        and c2["body"] < 0.35 * c2["range"]
        and c3["close"] > c3["open"]
        and c3["close"] > midpoint_c1
    )


def detect_gap_up(df: pd.DataFrame) -> bool:
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    return bool(float(curr["Open"]) > float(prev["High"]))


def detect_consolidation_breakout(df: pd.DataFrame) -> bool:
    if len(df) < 12:
        return False
    base = df.iloc[-11:-1]
    breakout = df.iloc[-1]
    tight_range_pct = ((float(base["High"].max()) - float(base["Low"].min())) / float(breakout["Close"])) * 100
    return bool(
        tight_range_pct < 5.0
        and float(breakout["Close"]) > float(base["High"].max())
        and float(breakout["Volume"]) > float(df["VOL20"].iloc[-1])
    )


def detect_resistance_breakout(df: pd.DataFrame) -> bool:
    if len(df) < 22:
        return False
    prev_high = float(df["High"].iloc[-21:-1].max())
    return bool(float(df["Close"].iloc[-1]) > prev_high * 1.002)


def detect_52w_high_proximity(df: pd.DataFrame) -> bool:
    high_52w = float(df["High"].max())
    close = float(df["Close"].iloc[-1])
    return bool(close >= high_52w * 0.95)


def evaluate_layer5(df: pd.DataFrame, layer2: Dict[str, Any]) -> Dict[str, Any]:
    patterns: List[str] = []
    triggers: List[str] = []

    if detect_hammer(df):
        patterns.append("Hammer")
        triggers.append("Hammer pattern suggests demand absorption")
    if detect_bullish_engulfing(df):
        patterns.append("Bullish Engulfing")
        triggers.append("Bullish engulfing reversal detected")
    if detect_morning_star(df):
        patterns.append("Morning Star")
        triggers.append("Morning star reversal sequence detected")
    if detect_gap_up(df) and layer2["conditions"].get("Volume surge above 1.5x 20D average", False):
        triggers.append("Gap-up with volume confirmation")
    if detect_consolidation_breakout(df):
        triggers.append("10-day consolidation breakout")
    if detect_resistance_breakout(df):
        triggers.append("Resistance breakout above prior swing highs")
    if detect_52w_high_proximity(df):
        triggers.append("Trading within 5% of 52-week high")

    catalyst_score = min(100.0, len(triggers) * 18 + len(patterns) * 8)
    return {
        "patterns": patterns,
        "triggers": triggers,
        "score": round(catalyst_score, 2),
    }


def compute_trade_levels(price: float, atr: float, config: Dict[str, Any]) -> Dict[str, float]:
    trade_cfg = config["trade"]
    raw_risk_per_share = atr * float(trade_cfg["atr_multiplier"])
    raw_stop_pct = raw_risk_per_share / price if price else 0.0
    stop_pct = clamp(
        raw_stop_pct,
        float(trade_cfg.get("min_sl_percent", 0.005)),
        float(trade_cfg.get("max_sl_percent", 0.05)),
    )
    risk_per_share = price * stop_pct
    stop_loss = price - risk_per_share
    target1 = price + risk_per_share * float(trade_cfg.get("target1_rr", 1.5))
    target2 = price + risk_per_share * float(trade_cfg.get("target2_rr", 2.5))
    target3 = price + risk_per_share * float(trade_cfg.get("target3_rr", 3.5))
    return {
        "entry": price,
        "risk_per_share": risk_per_share,
        "stop_loss": stop_loss,
        "stop_pct": stop_pct * 100,
        "target1": target1,
        "target2": target2,
        "target3": target3,
        "rr1": float(trade_cfg.get("target1_rr", 1.5)),
        "rr2": float(trade_cfg.get("target2_rr", 2.5)),
        "rr3": float(trade_cfg.get("target3_rr", 3.5)),
    }


def compute_position_size(price: float, risk_per_share: float, config: Dict[str, Any]) -> Tuple[int, float]:
    portfolio = float(config["portfolio"]["default_size"])
    risk_budget = portfolio * float(config["trade"]["risk_per_trade"])
    max_position_value = portfolio * float(config["trade"].get("max_position_percent", 0.20))

    if risk_per_share <= 0:
        return 0, 0.0

    qty_risk = int(risk_budget // risk_per_share)
    qty_cap = int(max_position_value // price) if price > 0 else 0
    qty = max(0, min(qty_risk, qty_cap))
    return qty, qty * price


def build_risk_flags(df: pd.DataFrame, layer3: Dict[str, Any], layer4: Dict[str, Any], layer5: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    latest = df.iloc[-1]

    if layer3["available_checks"] < 2:
        flags.append("Limited fundamentals from Yahoo Finance")
    if layer4["atr_pct"] > 6.0:
        flags.append("ATR elevated; volatility may widen stop-outs")
    if layer4["atr_pct"] < 1.5:
        flags.append("ATR muted; upside may take longer to develop")
    if layer4["five_day_return"] < 0:
        flags.append("5-day return still negative")
    if layer4["roc10"] < 0:
        flags.append("10-day ROC remains negative")
    if float(latest["RSI14"]) > 68:
        flags.append("RSI close to overbought zone")
    if not layer5["triggers"]:
        flags.append("No fresh catalyst or breakout pattern detected")
    if layer3["raw"].get("debt_to_equity") is not None and layer3["raw"]["debt_to_equity"] >= 1.5:
        flags.append("Leverage is elevated")
    if layer3["raw"].get("revenue_growth_pct") is not None and layer3["raw"]["revenue_growth_pct"] <= 0:
        flags.append("Revenue growth is not positive")
    return flags


def compute_scores(layer2: Dict[str, Any], layer3: Dict[str, Any], layer4: Dict[str, Any], layer5: Dict[str, Any], config: Dict[str, Any]) -> Tuple[float, float, float]:
    technical_score = round(
        (layer2["score"] * 0.70) + (layer4["score"] * 0.20) + (layer5["score"] * 0.10),
        2,
    )
    fundamental_score = round(layer3["score"], 2)
    setup_score = round((layer4["score"] * 0.65) + (layer5["score"] * 0.35), 2)

    tw = float(config["scoring"].get("technical_weight", 0.40))
    fw = float(config["scoring"].get("fundamental_weight", 0.30))
    remainder = max(0.0, 1.0 - tw - fw)
    confidence = round((technical_score * tw) + (fundamental_score * fw) + (setup_score * remainder), 2)
    return technical_score, fundamental_score, confidence


def estimate_return_range(entry: float, target1: float, target3: float) -> Tuple[float, float]:
    """FIX #1: report the TRUE return to target1 and target3, correctly ordered.

    The previous implementation applied `max(3.0, r1)` / `min(10.0, r3)`, which
    (a) floored the low end and capped the high end regardless of the actual
    targets, producing inverted bands like '3.00%-1.75%' for low-volatility
    names, and (b) silently hid real upside above 10%. The 3-10% intent is now
    enforced upstream as a selection filter (expected-move band in Layer 4), so
    this function does pure, honest reporting of the computed target returns.
    """
    r1 = ((target1 - entry) / entry) * 100 if entry else 0.0
    r3 = ((target3 - entry) / entry) * 100 if entry else 0.0
    lo, hi = sorted((round(r1, 2), round(r3, 2)))
    return lo, hi


def get_company_name(info: Dict[str, Any], symbol: str) -> str:
    return str(info.get("longName") or info.get("shortName") or symbol).strip()


def scan_symbol(symbol_row: pd.Series, config: Dict[str, Any], drop_partial: bool = True) -> Optional[StockResult]:
    symbol = str(symbol_row["Symbol"])
    fallback_sector = str(symbol_row.get("Sector", "Default"))
    history, info = fetch_stock_data(symbol, drop_partial=drop_partial)
    if len(history) < int(config["screener"].get("min_history_days", 200)):
        raise ScanError("Insufficient history to run 1-year momentum scan")

    layer1 = evaluate_layer1(history, config)
    if not layer1["passed"]:
        LOGGER.debug(
            "%s skipped by Layer 1: price=%s avg_volume_20d=%s",
            symbol,
            layer1["price"],
            layer1["avg_volume_20d"],
        )
        return None

    history = add_indicators(history)
    validate_indicator_snapshot(history)

    layer2 = evaluate_layer2(history, config)
    if not layer2["passed"]:
        return None

    layer3 = evaluate_layer3(info, fallback_sector, config)
    if not layer3["passed"]:
        return None

    layer4 = evaluate_layer4(history, layer2, config)

    # FIX #3: enforce the 3-10% expected-move band as an actual selection filter.
    if bool(config["trade"].get("enforce_return_band", True)) and not layer4["in_band"]:
        LOGGER.debug(
            "%s skipped by return band: expected %d-session move=%.2f%% (band %.1f-%.1f%%)",
            symbol,
            layer4["sessions"],
            layer4["expected_move_pct"],
            layer4["band_min"],
            layer4["band_max"],
        )
        return None

    layer5 = evaluate_layer5(history, layer2)

    latest = history.iloc[-1]
    technical_score, fundamental_score, confidence_score = compute_scores(layer2, layer3, layer4, layer5, config)
    trade = compute_trade_levels(float(latest["Close"]), layer4["atr"], config)
    position_qty, position_value = compute_position_size(trade["entry"], trade["risk_per_share"], config)
    risk_flags = build_risk_flags(history, layer3, layer4, layer5)
    return_min, return_max = estimate_return_range(trade["entry"], trade["target1"], trade["target3"])

    trigger_events = []
    trigger_events.extend(layer2["triggers"])
    trigger_events.extend(layer4["triggers"])
    trigger_events.extend(layer5["triggers"])

    deduped_triggers: List[str] = []
    for event in trigger_events:
        if event not in deduped_triggers:
            deduped_triggers.append(event)

    result = StockResult(
        symbol=symbol,
        company=get_company_name(info, symbol),
        sector=layer3["sector"],
        current_price=round(float(latest["Close"]), 2),
        technical_score=technical_score,
        fundamental_score=fundamental_score,
        setup_score=round((layer4["score"] * 0.65) + (layer5["score"] * 0.35), 2),
        confidence_score=confidence_score,
        entry=round(trade["entry"], 2),
        stop_loss=round(trade["stop_loss"], 2),
        stop_loss_pct=round(trade["stop_pct"], 2),
        target1=round(trade["target1"], 2),
        target2=round(trade["target2"], 2),
        target3=round(trade["target3"], 2),
        rr1=round(trade["rr1"], 2),
        rr2=round(trade["rr2"], 2),
        rr3=round(trade["rr3"], 2),
        return_min=return_min,
        return_max=return_max,
        expected_move_pct=round(layer4["expected_move_pct"], 2),  # FIX #3
        position_qty=position_qty,
        position_value=round(position_value, 2),
        validity_days=int(config["trade"]["validity_trading_days"]),
        avg_volume_20d=round(layer1["avg_volume_20d"], 2),
        atr=round(layer4["atr"], 2),
        atr_pct=round(layer4["atr_pct"], 2),
        rsi=round(float(latest["RSI14"]), 2),
        adx=round(float(latest["ADX14"]), 2),
        five_day_return=round(layer4["five_day_return"], 2),
        roc10=round(layer4["roc10"], 2),
        bb_position=round(layer4["bb_position"], 3),
        technical_pass_count=layer2["pass_count"],
        technical_condition_count=len(layer2["conditions"]),
        fundamental_pass_count=layer3["pass_count"],
        fundamental_condition_count=len(layer3["conditions"]),
        trigger_events=deduped_triggers,
        bullish_patterns=layer5["patterns"],
        risk_flags=risk_flags,
        technical_conditions=layer2["conditions"],
        fundamental_conditions=layer3["conditions"],
        raw_fundamentals=layer3["raw"],
    )
    return result


def save_results(results: List[StockResult], output_dir: Path, generated_at: datetime) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = generated_at.strftime("screener_%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"

    csv_records = [result.to_csv_record() for result in results]
    csv_columns = [
        "Symbol",
        "Company",
        "Price",
        "Sector",
        "TechnicalScore",
        "FundamentalScore",
        "ConfidenceScore",
        "Entry",
        "StopLoss",
        "Target1",
        "Target2",
        "Target3",
        "ExpectedMovePct",
        "ReturnToT1Pct",
        "ReturnToT3Pct",
        "TriggerEvents",
        "RiskFlags",
    ]
    pd.DataFrame(csv_records, columns=csv_columns).to_csv(csv_path, index=False)

    json_payload = {
        "generated_at": generated_at.isoformat(),
        "count": len(results),
        "results": [result.to_json_record() for result in results],
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    return csv_path, json_path


def render_summary_table(results: List[StockResult]) -> None:
    table = Table(title="Shortlisted NSE Stocks Under ₹300", box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Symbol", style="cyan")
    table.add_column("Price", justify="right")
    table.add_column("Sector")
    table.add_column("Tech", justify="right")
    table.add_column("Fund", justify="right")
    table.add_column("Conf.", justify="right", style="bold green")
    table.add_column("Exp. Move", justify="right")
    table.add_column("Return to T1-T3", justify="right")
    table.add_column("Top Trigger")

    for index, result in enumerate(results, start=1):
        trigger = result.trigger_events[0] if result.trigger_events else "Momentum setup"
        table.add_row(
            str(index),
            result.symbol,
            rupees(result.current_price),
            result.sector,
            f"{result.technical_score:.1f}",
            f"{result.fundamental_score:.1f}",
            f"{result.confidence_score:.1f}",
            f"{result.expected_move_pct:.1f}%",
            f"{result.return_min:.1f}%–{result.return_max:.1f}%",
            trigger,
        )
    CONSOLE.print(table)


def render_detail_card(result: StockResult, portfolio_size: float, risk_per_trade: float) -> None:
    detail = Table.grid(padding=(0, 1))
    detail.add_column(style="cyan", width=30)
    detail.add_column(style="white")

    detail.add_row("Stock Info", f"{result.symbol} | {result.company} | {rupees(result.current_price)} | {result.sector}")
    detail.add_row("Expected Move (band gate)", f"{result.expected_move_pct:.2f}% over {result.validity_days} sessions")
    detail.add_row("Return to Targets", f"T1 {result.return_min:.2f}% … T3 {result.return_max:.2f}%")
    detail.add_row("Trigger Events", ", ".join(result.trigger_events[:6]) or "None")
    detail.add_row("Entry Level", rupees(result.entry))
    detail.add_row("🎯 Target 1 (1.5:1)", f"{rupees(result.target1)} | R:R {result.rr1:.1f}:1")
    detail.add_row("🎯 Target 2 (2.5:1)", f"{rupees(result.target2)} | R:R {result.rr2:.1f}:1")
    detail.add_row("🎯 Target 3 (3.5:1)", f"{rupees(result.target3)} | R:R {result.rr3:.1f}:1")
    detail.add_row("Stop Loss", f"{rupees(result.stop_loss)} ({result.stop_loss_pct:.2f}%)")
    detail.add_row("Technical Score", f"{result.technical_score:.1f}/100 | RSI {result.rsi:.1f} | ADX {result.adx:.1f}")
    detail.add_row("Fundamental Score", f"{result.fundamental_score:.1f}/100")
    detail.add_row("Overall Confidence", f"{result.confidence_score:.1f}/100")
    detail.add_row("Bullish Patterns", ", ".join(result.bullish_patterns) or "None")
    detail.add_row("Risk Flags", ", ".join(result.risk_flags) or "None")
    detail.add_row(
        "Position Sizing",
        f"{result.position_qty} shares (~{rupees(result.position_value)}) for {rupees(portfolio_size)} portfolio at {risk_per_trade * 100:.1f}% risk/trade",
    )
    detail.add_row("Validity", f"Signal valid for next {result.validity_days} trading sessions")

    header = f"[bold green]{result.symbol}[/bold green] • Confidence {result.confidence_score:.1f}/100"
    CONSOLE.print(Panel(detail, title=header, border_style="green", box=box.ROUNDED))


def render_zero_state(output_dir: Path) -> None:
    message = (
        "No stocks met all shortlist criteria in this run. This can happen when price, volume, trend, "
        "fundamentals, or the expected-move band are not satisfied. "
        f"Try a smaller --limit for debugging, --no-return-band to relax the 3-10%% filter, "
        f"or adjust config thresholds in {output_dir.parent / 'config.yaml' if output_dir.parent else 'config.yaml'}."
    )
    CONSOLE.print(Panel(message, title="No Qualified Stocks", border_style="red", box=box.ROUNDED))


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    config = load_config(args.config, args)
    generated_at = datetime.now()
    drop_partial = not getattr(args, "keep_incomplete_bar", False)  # FIX #2

    print_disclaimer()
    universe_path = Path("nse_stocks.csv")
    universe = load_universe(universe_path, args.limit)
    CONSOLE.print(f"[bold cyan]Universe loaded:[/bold cyan] {len(universe)} stocks")
    band_state = "ON" if config["trade"].get("enforce_return_band", True) else "OFF"
    CONSOLE.print(
        f"[bold cyan]Scan filters:[/bold cyan] ₹{config['screener']['min_price']:.0f} to ₹{config['screener']['max_price']:.0f}, "
        f"20D avg volume > {int(config['screener']['min_avg_volume_20d']):,} | "
        f"return band {config['trade']['return_band_min']:.0f}-{config['trade']['return_band_max']:.0f}% [{band_state}] | "
        f"partial-bar drop {'ON' if drop_partial else 'OFF'}"
    )

    results: List[StockResult] = []
    start_time = time.time()
    portfolio_size = float(config["portfolio"]["default_size"])
    risk_per_trade = float(config["trade"]["risk_per_trade"])

    for index, (_, row) in enumerate(universe.iterrows(), start=1):
        symbol = str(row["Symbol"])
        CONSOLE.print(f"[blue]Scanning {index}/{len(universe)}:[/blue] {symbol}")
        try:
            result = scan_symbol(row, config, drop_partial=drop_partial)
            if result is not None:
                results.append(result)
                LOGGER.info(
                    "%s shortlisted | price=%s | confidence=%.2f | exp_move=%.2f%%",
                    result.symbol,
                    result.current_price,
                    result.confidence_score,
                    result.expected_move_pct,
                )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if args.verbose:
                LOGGER.exception("Failed to process %s: %s", symbol, exc)
            else:
                LOGGER.warning("Failed to process %s: %s", symbol, exc)
        time.sleep(0.05)

    results.sort(key=lambda item: (item.confidence_score, item.technical_score, item.fundamental_score), reverse=True)
    max_results = int(config.get("scoring", {}).get("max_buy_signals", 5))
    results = results[:max_results]

    output_dir = Path(args.output_dir)
    csv_path, json_path = save_results(results, output_dir, generated_at)

    CONSOLE.print()
    if results:
        render_summary_table(results)
        CONSOLE.print()
        for result in results:
            render_detail_card(result, portfolio_size, risk_per_trade)
    else:
        render_zero_state(Path(args.config))

    elapsed = time.time() - start_time
    CONSOLE.print(
        Panel(
            f"Completed in {elapsed:.1f}s\nCSV: {csv_path}\nJSON: {json_path}",
            title="Run Complete",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        CONSOLE.print("[red]Interrupted by user[/red]")
        raise SystemExit(130)
