"""
Implementación de Indicadores Técnicos
=======================================
Basado en fórmulas estándar de TradingView y Pine Script.
"""

import pandas as pd
import numpy as np




def ema(series, length):
    """Exponential Moving Average"""
    return series.ewm(span=length, adjust=False).mean()

def sma(series, length):
    """Simple Moving Average"""
    return series.rolling(length).mean()

def wma(series, length):
    """Weighted Moving Average (mayor peso a precios más recientes)"""
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def hma(series, length):
    """Hull Moving Average (reduce lag)"""
    half = int(length / 2)
    sqrt_len = int(np.sqrt(length))
    wma_half = wma(series, half)
    wma_full = wma(series, length)
    hull_raw = 2 * wma_half - wma_full
    return wma(hull_raw, sqrt_len)

def dema(series, length):
    """Double Exponential Moving Average"""
    e = ema(series, length)
    return 2 * e - ema(e, length)

def ma(series, ma_type, length):
    """Selector de tipo de media móvil"""
    movers = {
        "EMA": ema, "SMA": sma, "WMA": wma,
        "HMA": hma, "DEMA": dema
    }
    return movers.get(ma_type, ema)(series, length)

def rma(series, length):
    """Wilder's Moving Average (usado en RSI original)"""
    return series.ewm(alpha=1/length, adjust=False).mean()

def rsi_tv(series, length):
    """Relative Strength Index (versión TradingView)"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def adx_tv(high, low, close, length):
    """Average Directional Index (TradingView versión)"""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    tr_rma = rma(tr, length)

    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    plus_dm_rma = rma(pd.Series(plus_dm, index=high.index), length)
    minus_dm_rma = rma(pd.Series(minus_dm, index=high.index), length)

    plus_di = 100 * (plus_dm_rma / tr_rma)
    minus_di = 100 * (minus_dm_rma / tr_rma)

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = rma(dx, length)

    return adx, plus_di, minus_di

