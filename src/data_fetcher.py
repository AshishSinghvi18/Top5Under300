"""Data fetching module — yfinance, NSE endpoints, RSS feeds with diskcache."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import feedparser
import pandas as pd
import yaml
import yfinance as yf
from diskcache import Cache
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_cache(config: Dict[str, Any]) -> Cache:
    cache_dir = config.get("performance", {}).get("cache_dir", ".cache/screener")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return Cache(cache_dir)


def _retry_attempts(config: Dict[str, Any]) -> int:
    return int(config.get("performance", {}).get("retry_max_attempts", 3))


def _call_timeout(config: Dict[str, Any]) -> int:
    return int(config.get("performance", {}).get("call_timeout_seconds", 30))


# ---------------------------------------------------------------------------
# Universe loading
# ---------------------------------------------------------------------------

_FALLBACK_SYMBOLS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
    "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "TITAN",
    "SUNPHARMA", "HCLTECH", "WIPRO", "TATAMOTORS", "NTPC",
]


def load_nse_universe(
    config: Dict[str, Any],
    static_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Load NSE stock universe from a CSV file or use a hardcoded fallback."""
    if static_csv and os.path.isfile(static_csv):
        try:
            df = pd.read_csv(static_csv)
            if "Symbol" not in df.columns:
                logger.warning("CSV missing 'Symbol' column, using fallback universe")
            else:
                if "Series" not in df.columns:
                    df["Series"] = "EQ"
                logger.info(f"Loaded {len(df)} stocks from {static_csv}")
                return df
        except Exception as exc:
            logger.warning(f"Failed to read {static_csv}: {exc}")

    logger.warning("Using hardcoded fallback universe (~20 stocks)")
    return pd.DataFrame({
        "Symbol": _FALLBACK_SYMBOLS,
        "Series": ["EQ"] * len(_FALLBACK_SYMBOLS),
    })


# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    symbol: str,
    config: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data for *symbol* via yfinance, with diskcache caching."""
    min_days = int(config.get("screener", {}).get("min_history_days", 200))
    cache_hours = int(config.get("performance", {}).get("price_cache_hours", 24))
    cache_key = f"ohlcv:{symbol}"

    cache = _get_cache(config)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        # Fetch extra buffer to ensure we have enough trading days
        cal_days = int(min_days * 1.6) + 30
        df = ticker.history(period=f"{cal_days}d")
        if df is None or df.empty:
            logger.debug(f"{symbol}: no OHLCV data returned")
            return None

        # Normalise column names to standard yfinance names
        expected = ["Open", "High", "Low", "Close", "Volume"]
        for col in expected:
            if col not in df.columns:
                logger.debug(f"{symbol}: missing column {col}")
                return None

        df = df[expected].copy()
        df.dropna(subset=["Close"], inplace=True)

        if len(df) < min_days:
            logger.debug(f"{symbol}: only {len(df)} rows, need {min_days}")
            return None

        cache.set(cache_key, df, expire=cache_hours * 3600)
        return df
    except Exception as exc:
        logger.debug(f"{symbol}: OHLCV fetch error — {exc}")
        return None


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

def fetch_fundamentals(
    symbol: str,
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return fundamental data dict from yfinance .info for *symbol*."""
    cache_hours = int(config.get("performance", {}).get("price_cache_hours", 24))
    cache_key = f"fund:{symbol}"

    cache = _get_cache(config)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info or {}
        if not info or info.get("regularMarketPrice") is None:
            logger.debug(f"{symbol}: empty yfinance info")
            return None

        result: Dict[str, Any] = {
            "pe": info.get("trailingPE") or info.get("forwardPE"),
            "debt_to_equity": _safe_div(info.get("totalDebt"), info.get("totalStockholderEquity")),
            "roe": info.get("returnOnEquity"),
            "revenue_growth": info.get("revenueGrowth"),
            "sector": info.get("sector"),
            "long_name": info.get("longName") or info.get("shortName") or symbol,
            "market_cap": info.get("marketCap"),
        }
        cache.set(cache_key, result, expire=cache_hours * 3600)
        return result
    except Exception as exc:
        logger.debug(f"{symbol}: fundamentals fetch error — {exc}")
        return None


def _safe_div(a: Any, b: Any) -> Optional[float]:
    try:
        if a is not None and b is not None and b != 0:
            return float(a) / float(b)
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return None


# ---------------------------------------------------------------------------
# ASM / GSM surveillance list
# ---------------------------------------------------------------------------

def fetch_asm_gsm_list(config: Dict[str, Any]) -> Optional[set]:
    """Best-effort fetch of ASM/GSM surveillance symbols. Returns None on failure."""
    cache_hours = int(config.get("performance", {}).get("asm_cache_hours", 24))
    cache_key = "asm_gsm_set"

    cache = _get_cache(config)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        import requests

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        }
        url = "https://www.nseindia.com/api/reportASMGSM"
        timeout = _call_timeout(config)

        session = requests.Session()
        # Hit the main page first so we get cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=timeout)
        resp = session.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()

        data = resp.json()
        symbols: set = set()
        if isinstance(data, list):
            for item in data:
                sym = item.get("symbol") or item.get("Symbol")
                if sym:
                    symbols.add(str(sym).strip().upper())
        elif isinstance(data, dict):
            for key in ("ASM", "GSM", "data"):
                for item in data.get(key, []):
                    sym = item.get("symbol") or item.get("Symbol")
                    if sym:
                        symbols.add(str(sym).strip().upper())

        if symbols:
            cache.set(cache_key, symbols, expire=cache_hours * 3600)
            logger.info(f"Fetched {len(symbols)} ASM/GSM symbols")
            return symbols
    except Exception as exc:
        logger.warning(f"ASM/GSM fetch failed (best-effort): {exc}")

    return None


# ---------------------------------------------------------------------------
# Sector index returns
# ---------------------------------------------------------------------------

def fetch_sector_index_returns(
    sector_mapping_path: str,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Fetch 20-day returns for all sector index tickers in the mapping file."""
    try:
        with open(sector_mapping_path) as f:
            mapping = yaml.safe_load(f)
    except Exception as exc:
        logger.warning(f"Could not load sector mapping: {exc}")
        return {}

    tickers: set = set()
    for idx_ticker in mapping.get("sector_to_index", {}).values():
        tickers.add(idx_ticker)
    default = mapping.get("default_index")
    if default:
        tickers.add(default)

    cache = _get_cache(config)
    cache_hours = int(config.get("performance", {}).get("price_cache_hours", 24))
    results: Dict[str, float] = {}

    for ticker in tickers:
        cache_key = f"sector_ret:{ticker}"
        cached = cache.get(cache_key)
        if cached is not None:
            results[ticker] = cached
            continue
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="40d")
            if hist is not None and len(hist) >= 20:
                close = hist["Close"]
                ret = (float(close.iloc[-1]) - float(close.iloc[-20])) / float(close.iloc[-20])
                results[ticker] = ret
                cache.set(cache_key, ret, expire=cache_hours * 3600)
            else:
                logger.debug(f"Insufficient data for sector index {ticker}")
        except Exception as exc:
            logger.debug(f"Sector index {ticker} fetch error: {exc}")

    logger.info(f"Sector returns fetched for {len(results)}/{len(tickers)} indices")
    return results


# ---------------------------------------------------------------------------
# Corporate actions
# ---------------------------------------------------------------------------

def fetch_corporate_actions(
    symbol: str,
    config: Dict[str, Any],
    lookahead_days: int = 7,
) -> Optional[List[Dict[str, Any]]]:
    """Best-effort fetch of upcoming corporate actions for *symbol*."""
    cache_hours = int(config.get("performance", {}).get("price_cache_hours", 24))
    cache_key = f"corp:{symbol}:{lookahead_days}"

    cache = _get_cache(config)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        actions = ticker.actions
        if actions is None or actions.empty:
            result: List[Dict[str, Any]] = []
            cache.set(cache_key, result, expire=cache_hours * 3600)
            return result

        now = datetime.now()
        cutoff = now + timedelta(days=lookahead_days)
        upcoming: List[Dict[str, Any]] = []

        for dt_idx, row in actions.iterrows():
            action_date = pd.Timestamp(dt_idx)
            if action_date.tz is not None:
                action_date = action_date.tz_localize(None)
            if now <= action_date <= cutoff:
                upcoming.append({
                    "date": action_date.strftime("%Y-%m-%d"),
                    "dividends": float(row.get("Dividends", 0)),
                    "stock_splits": float(row.get("Stock Splits", 0)),
                })

        cache.set(cache_key, upcoming, expire=cache_hours * 3600)
        return upcoming
    except Exception as exc:
        logger.debug(f"{symbol}: corporate actions fetch error — {exc}")
        return None


# ---------------------------------------------------------------------------
# Promoter holding
# ---------------------------------------------------------------------------

def fetch_promoter_holding(
    symbol: str,
    config: Dict[str, Any],
) -> Optional[Dict[str, Optional[float]]]:
    """Best-effort fetch of promoter holding percentages."""
    cache_hours = int(config.get("performance", {}).get("price_cache_hours", 24))
    cache_key = f"promoter:{symbol}"

    cache = _get_cache(config)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        holders = ticker.major_holders
        if holders is None or holders.empty:
            return None

        current: Optional[float] = None
        previous: Optional[float] = None

        for _, row_data in holders.iterrows():
            label = str(row_data.iloc[-1]).lower() if len(row_data) > 1 else ""
            value = row_data.iloc[0]
            if "insider" in label or "promoter" in label:
                try:
                    val = float(str(value).replace("%", ""))
                    if current is None:
                        current = val
                    else:
                        previous = val
                except (ValueError, TypeError):
                    pass

        if current is not None:
            result = {"current": current, "previous": previous}
            cache.set(cache_key, result, expire=cache_hours * 3600)
            return result
    except Exception as exc:
        logger.debug(f"{symbol}: promoter holding fetch error — {exc}")

    return None


# ---------------------------------------------------------------------------
# RSS news
# ---------------------------------------------------------------------------

def fetch_rss_news(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch and aggregate RSS news from configured feeds."""
    feeds = config.get("sentiment", {}).get("rss_feeds", [])
    cache_hours = int(config.get("performance", {}).get("rss_cache_hours", 2))
    cache = _get_cache(config)

    all_items: List[Dict[str, Any]] = []

    for feed_url in feeds:
        cache_key = f"rss:{feed_url}"
        cached = cache.get(cache_key)
        if cached is not None:
            all_items.extend(cached)
            continue

        items: List[Dict[str, Any]] = []
        fetch_failed = False
        try:
            parsed = feedparser.parse(feed_url)
            if parsed.bozo and not parsed.entries:
                raise ValueError(f"Feed parse error: {parsed.bozo_exception}")

            for entry in parsed.entries:
                items.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "published": entry.get("published", ""),
                    "source": feed_url,
                    "fetch_failed": False,
                })
            cache.set(cache_key, items, expire=cache_hours * 3600)
            logger.debug(f"RSS: {len(items)} items from {feed_url}")
        except Exception as exc:
            logger.warning(f"RSS fetch failed for {feed_url}: {exc}")
            fetch_failed = True
            items = [{
                "title": "",
                "summary": "",
                "published": "",
                "source": feed_url,
                "fetch_failed": True,
            }]

        all_items.extend(items)

    return all_items


def all_rss_failed(news_items: List[Dict[str, Any]]) -> bool:
    """Return True if every news item has fetch_failed=True or list is empty."""
    if not news_items:
        return True
    return all(item.get("fetch_failed", False) for item in news_items)
