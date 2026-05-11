"""NSE trading-day arithmetic.

Uses ``pandas_market_calendars`` (exchange **NSE**) when available for
holiday-aware logic; falls back to simple weekday (Mon–Fri) heuristics
when the package is not installed.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

# IST is UTC+05:30 (fixed offset, no DST).
_IST = timezone(timedelta(hours=5, minutes=30))
_MARKET_OPEN = time(9, 15)
_MARKET_CLOSE = time(15, 30)

try:
    import pandas_market_calendars as mcal

    _nse_cal = mcal.get_calendar("NSE")
    _HAS_MCAL = True
except Exception:  # ImportError, or calendar not found
    _nse_cal = None
    _HAS_MCAL = False


# ── internal helpers ────────────────────────────────────────────────────


def _is_trading_day_mcal(d: date) -> bool:
    """Check via pandas_market_calendars whether *d* is a valid session."""
    schedule = _nse_cal.schedule(start_date=d.isoformat(), end_date=d.isoformat())  # type: ignore[union-attr]
    return len(schedule) > 0


def _is_trading_day_fallback(d: date) -> bool:
    """Simple weekday check (Mon=0 … Fri=4)."""
    return d.weekday() < 5


def _prev_trading_day(d: date, checker) -> date:
    """Walk backwards from *d* (exclusive) until a trading day is found."""
    candidate = d - timedelta(days=1)
    while not checker(candidate):
        candidate -= timedelta(days=1)
    return candidate


# ── public API ──────────────────────────────────────────────────────────


def is_trading_day(dt: date) -> bool:
    """Return ``True`` if *dt* is an NSE trading day."""
    if _HAS_MCAL:
        return _is_trading_day_mcal(dt)
    return _is_trading_day_fallback(dt)


def previous_trading_day(dt: date) -> date:
    """Return the trading day immediately before *dt*."""
    checker = _is_trading_day_mcal if _HAS_MCAL else _is_trading_day_fallback
    return _prev_trading_day(dt, checker)


def latest_trading_day(dt: datetime) -> date:
    """Return the most recent NSE trading day on or before *dt*.

    Rules
    -----
    * Convert *dt* to IST for comparison with market hours.
    * **Weekday, on or after market open (09:15 IST)** → return that date
      (covers both intra-day and post-close scenarios).
    * **Weekday, before market open (09:15 IST)** → return the previous
      trading day.
    * **Weekend / holiday** → return the most recent prior trading day.
    """
    # Ensure we work in IST regardless of the caller's timezone.
    if dt.tzinfo is None:
        ist_dt = dt.replace(tzinfo=_IST)
    else:
        ist_dt = dt.astimezone(_IST)

    today = ist_dt.date()
    current_time = ist_dt.time()  # strip tz for comparison with naive times

    checker = _is_trading_day_mcal if _HAS_MCAL else _is_trading_day_fallback

    if checker(today):
        if current_time >= _MARKET_OPEN:
            return today
        return _prev_trading_day(today, checker)

    # Weekend or holiday → walk back from today (exclusive).
    return _prev_trading_day(today, checker)
