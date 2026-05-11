"""Technical indicator calculation functions for NSE stock screening.

Pure math functions using only pandas and numpy — no external TA libraries.
"""

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EMA with alpha=1/period)."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(
    close: pd.Series, fast: int, slow: int, signal: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_sma(close: pd.Series, period: int) -> pd.Series:
    """Compute simple moving average."""
    return close.rolling(window=period).mean()


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    """Compute Average True Range using Wilder's smoothing."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    """Compute Average Directional Index (ADX)."""
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    atr = compute_atr(high, low, close, period)

    smooth_plus_dm = plus_dm.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    smooth_minus_dm = minus_dm.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()

    plus_di = 100 * smooth_plus_dm / atr
    minus_di = 100 * smooth_minus_dm / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx


def compute_supertrend(
    df: pd.DataFrame, period: int, multiplier: float
) -> pd.DataFrame:
    """Compute Supertrend indicator.

    Returns a DataFrame with columns:
        supertrend — the supertrend price level
        direction  — 1 (bullish / up) or -1 (bearish / down)
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    atr = compute_atr(high, low, close, period)
    hl2 = (high + low) / 2

    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    n = len(df)
    upper_band = np.empty(n, dtype=np.float64)
    lower_band = np.empty(n, dtype=np.float64)
    supertrend = np.empty(n, dtype=np.float64)
    direction = np.empty(n, dtype=np.float64)

    upper_band[:] = np.nan
    lower_band[:] = np.nan
    supertrend[:] = np.nan
    direction[:] = np.nan

    bu = basic_upper.values
    bl = basic_lower.values
    cl = close.values

    for i in range(n):
        if np.isnan(bu[i]) or np.isnan(bl[i]):
            continue

        # Final upper band
        if i == 0 or np.isnan(upper_band[i - 1]):
            upper_band[i] = bu[i]
        else:
            upper_band[i] = (
                min(bu[i], upper_band[i - 1])
                if cl[i - 1] <= upper_band[i - 1]
                else bu[i]
            )

        # Final lower band
        if i == 0 or np.isnan(lower_band[i - 1]):
            lower_band[i] = bl[i]
        else:
            lower_band[i] = (
                max(bl[i], lower_band[i - 1])
                if cl[i - 1] >= lower_band[i - 1]
                else bl[i]
            )

        # Direction and supertrend value
        if i == 0 or np.isnan(supertrend[i - 1]):
            direction[i] = 1 if cl[i] > upper_band[i] else -1
        elif supertrend[i - 1] == upper_band[i - 1]:
            # Previous trend was bearish (tracking upper band)
            direction[i] = 1 if cl[i] > upper_band[i] else -1
        else:
            # Previous trend was bullish (tracking lower band)
            direction[i] = -1 if cl[i] < lower_band[i] else 1

        supertrend[i] = (
            lower_band[i] if direction[i] == 1 else upper_band[i]
        )

    direction_series = pd.array(direction, dtype=pd.Int32Dtype())
    result = pd.DataFrame(
        {"supertrend": supertrend, "direction": direction_series},
        index=df.index,
    )
    return result
