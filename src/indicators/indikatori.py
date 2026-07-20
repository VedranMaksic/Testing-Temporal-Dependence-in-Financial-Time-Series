from __future__ import annotations
import numpy as np
import pandas as pd


# =====================================================
# ================= CORE INDICATORS ===================
# =====================================================

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def roc(series: pd.Series, period: int) -> pd.Series:
    return series.pct_change(periods=period) * 100.0


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14):
    return rsi_wilder(close, period)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    return atr_wilder(high, low, close, period)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume.fillna(0)).cumsum()


# =====================================================
# ================= TREND FEATURES ====================
# =====================================================

def sma_gt(sma_fast, sma_slow):
    return (sma_fast > sma_slow).astype(int)


def sma_cross(sma_fast, sma_slow):
    signal = (sma_fast > sma_slow).astype(int)
    return signal.diff().fillna(0).clip(-1, 1)


def spread(a, b):
    return a - b


def dist_rel(price, ma):
    return (price - ma) / ma


def trend_strength(ema20, sma50, atr14):
    return (ema20 - sma50).abs() / atr14.replace(0, np.nan)


# =====================================================
# ================= RSI FEATURES ======================
# =====================================================

def rsi_overbought(rsi):
    return (rsi > 70).astype(int)


def rsi_oversold(rsi):
    return (rsi < 30).astype(int)


def rsi_gt_50(rsi):
    return (rsi > 50).astype(int)


def rsi_cross_50(rsi):
    signal = (rsi > 50).astype(int)
    return signal.diff().fillna(0).clip(-1, 1)


def rsi_centered(rsi):
    return rsi - 50


def rsi_abs_dist_50(rsi):
    return (rsi - 50).abs()


def rsi_regime_3(rsi):
    return np.select(
        [rsi < 30, rsi > 70],
        [0, 2],
        default=1
    ).astype(float)


def rsi_ema(rsi, period: int = 10):
    return rsi.ewm(span=period, adjust=False).mean()


# =====================================================
# ================= ATR FEATURES ======================
# =====================================================

def atr_pct(high, low, close, period: int = 14):
    return atr(high, low, close, period) / close


def atr_pct_zscore(atr_pct_series, period: int = 60):
    mean = atr_pct_series.rolling(period, min_periods=period).mean()
    std = atr_pct_series.rolling(period, min_periods=period).std()
    return (atr_pct_series - mean) / std


# =====================================================
# ================= OBV FEATURES ======================
# =====================================================

def obv_slope(obv):
    return obv.diff()


def obv_slope_n(obv, period: int = 5):
    return obv.diff(period)


def obv_slope_over_atr(obv_slope, atr14):
    return obv_slope / atr14.replace(0, np.nan)


def obv_div_bear(close, obv):
    return ((close.diff(5) > 0) & (obv.diff(5) < 0)).astype(int)


def obv_div_bull(close, obv):
    return ((close.diff(5) < 0) & (obv.diff(5) > 0)).astype(int)


# =====================================================
# ================= PRICE CONTEXT =====================
# =====================================================

def log_price(price):
    return np.log(price.where(price > 0))


def zscore(series, period: int = 60):
    mean = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std()
    return (series - mean) / std


def dist_from_high(price, period: int = 60):
    return (price / price.rolling(period, min_periods=period).max()) - 1.0


def dist_from_low(price, period: int = 60):
    return (price / price.rolling(period, min_periods=period).min()) - 1.0


# =====================================================
# ================= RETURNS / VOL =====================
# =====================================================

def log_return(price, period: int = 1):
    return np.log(price).diff(period)


def rolling_volatility(log_ret, period: int = 20):
    return log_ret.rolling(period, min_periods=period).std()


def vol_ratio(vol_short, vol_long):
    return vol_short / vol_long


# =====================================================
# ================= VOLUME FEATURES ===================
# =====================================================

def volume_zscore(volume, period: int = 20):
    mean = volume.rolling(period, min_periods=period).mean()
    std = volume.rolling(period, min_periods=period).std()
    return (volume - mean) / std


def volume_ratio_short_long(volume, short: int = 5, long: int = 20):
    short_ma = volume.rolling(short, min_periods=short).mean()
    long_ma = volume.rolling(long, min_periods=long).mean()
    return short_ma / long_ma


# =====================================================
# ================= PRICE VS MA =======================
# =====================================================

def price_above(price, ma):
    return (price > ma).astype(float)


def slope(series, period: int = 10):
    return series.diff(period)

##novo
def rsi_div_bear(close, rsi, period: int = 5):
    price_up = close.diff(period) > 0
    rsi_down = rsi.diff(period) < 0
    return (price_up & rsi_down).astype(int)


def rsi_div_bull(close, rsi, period: int = 5):
    price_down = close.diff(period) < 0
    rsi_up = rsi.diff(period) > 0
    return (price_down & rsi_up).astype(int)

def momentum_acceleration(series, short: int = 5, long: int = 20):
    short_mom = series.diff(short)
    long_mom = series.diff(long)
    return short_mom - long_mom

def volatility_compression(log_ret, short: int = 10, long: int = 50):
    vol_short = log_ret.rolling(short, min_periods=short).std()
    vol_long = log_ret.rolling(long, min_periods=long).std()
    return vol_short / vol_long

def volatility_compression(log_ret, short: int = 10, long: int = 50):
    vol_short = log_ret.rolling(short, min_periods=short).std()
    vol_long = log_ret.rolling(long, min_periods=long).std()
    return vol_short / vol_long

def true_range(high, low, close):
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), 
         (high - prev_close).abs(), 
         (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr


def range_expansion(high, low, close, period: int = 20):
    tr = true_range(high, low, close)
    return tr / tr.rolling(period, min_periods=period).mean()

def ema_slope(ema_series, period: int = 5):
    return ema_series.diff(period)