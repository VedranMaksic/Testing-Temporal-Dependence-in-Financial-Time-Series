from dataclasses import dataclass


@dataclass
class Strategy:
    name: str = "up5_daily_enhanced_all"
    timeframe: str = "1d"

    instruments = [
        # =========================
    # CRYPTO
    # =========================
    "BTC-USD",
    "ETH-USD",
    "BNB-USD",
    "SOL-USD",
    "XRP-USD",
    "ADA-USD",
    "DOGE-USD",

    # =========================
    # METALS
    # =========================
    "GC=F",
    "SI=F",
    "HG=F",

    # =========================
    # BONDS
    # =========================
    "IEF",
    "TLT",
    "SHY",

    # =========================
    # STOCKS (TECH / BIG CAP)
    # =========================
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",

    # =========================
    # INDICES
    # =========================
    "^GSPC",
    "^NDX",
    "^DJI",
    "^RUT",

    # =========================
    # FX
    # =========================
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    ]

    #time split
    split_date: str = "2023-01-01"

    # Target definicija
    target = {
        "module": "up_down",
        "params": {
            "direction": "up",
            "horizon": 60,
            "threshold": 0.05
        }
    }

    # Model definicija
    model = {
        "module": "xgboost_trainer",
        "params": {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05
        }
    }


    # Feature definicija (feature_name: (function_name, params))
    features = {

    # =====================================================
    # ================= BASIC TREND =======================
    # =====================================================

    "SMA_10": {
        "func": "sma",
        "args": ["Close"],
        "kwargs": {"period": 10}
    },

    "SMA_50": {
        "func": "sma",
        "args": ["Close"],
        "kwargs": {"period": 50}
    },

    "EMA_20": {
        "func": "ema",
        "args": ["Close"],
        "kwargs": {"period": 20}
    },

    "SMA_CROSS_10_50": {
        "func": "sma_cross",
        "args": ["SMA_10", "SMA_50"],
        "kwargs": {}
    },
    "ATR_14": { #added here because trend strength is next
        "func": "atr",
        "args": ["High", "Low", "Close"],
        "kwargs": {"period": 14}
    },

    "TREND_STRENGTH": {
        "func": "trend_strength",
        "args": ["EMA_20", "SMA_50", "ATR_14"],
        "kwargs": {}
    },

    # =====================================================
    # ================= MOMENTUM ==========================
    # =====================================================

    "RSI_14": {
        "func": "rsi",
        "args": ["Close"],
        "kwargs": {"period": 14}
    },

    "RSI_CENTERED": {
        "func": "rsi_centered",
        "args": ["RSI_14"],
        "kwargs": {}
    },

    "RSI_REGIME_3": {
        "func": "rsi_regime_3",
        "args": ["RSI_14"],
        "kwargs": {}
    },

    "RSI_CROSS_50": {
        "func": "rsi_cross_50",
        "args": ["RSI_14"],
        "kwargs": {}
    },

    "RSI_DIV_BULL": {
        "func": "rsi_div_bull",
        "args": ["Close", "RSI_14"],
        "kwargs": {"period": 5}
    },

    "RSI_DIV_BEAR": {
        "func": "rsi_div_bear",
        "args": ["Close", "RSI_14"],
        "kwargs": {"period": 5}
    },

    # =====================================================
    # ================= VOLATILITY ========================
    # =====================================================

    

    "ATR_PCT": {
        "func": "atr_pct",
        "args": ["High", "Low", "Close"],
        "kwargs": {"period": 14}
    },

    "ATR_PCT_Z": {
        "func": "atr_pct_zscore",
        "args": ["ATR_PCT"],
        "kwargs": {"period": 60}
    },

    "LOG_RETURN_1": {
        "func": "log_return",
        "args": ["Close"],
        "kwargs": {"period": 1}
    },

    "ROLLING_VOL_20": {
        "func": "rolling_volatility",
        "args": ["LOG_RETURN_1"],
        "kwargs": {"period": 20}
    },

    "VOL_COMPRESSION": {
        "func": "volatility_compression",
        "args": ["LOG_RETURN_1"],
        "kwargs": {"short": 10, "long": 50}
    },

    # =====================================================
    # ================= VOLUME ============================
    # =====================================================

    "OBV": {
        "func": "obv",
        "args": ["Close", "Volume"],
        "kwargs": {}
    },

    "OBV_SLOPE_5": {
        "func": "obv_slope_n",
        "args": ["OBV"],
        "kwargs": {"period": 5}
    },

    "OBV_DIV_BULL": {
        "func": "obv_div_bull",
        "args": ["Close", "OBV"],
        "kwargs": {}
    },

    "OBV_DIV_BEAR": {
        "func": "obv_div_bear",
        "args": ["Close", "OBV"],
        "kwargs": {}
    },

    "VOLUME_Z": {
        "func": "volume_zscore",
        "args": ["Volume"],
        "kwargs": {"period": 20}
    },

    # =====================================================
    # ================= BREAKOUT / STRUCTURE ==============
    # =====================================================

    # "DIST_FROM_HIGH_60": {
    #     "func": "dist_from_high",
    #     "args": ["Close"],
    #     "kwargs": {"period": 60}
    # },

    # "DIST_FROM_LOW_60": {
    #     "func": "dist_from_low",
    #     "args": ["Close"],
    #     "kwargs": {"period": 60}
    # },

    

    # "RANGE_EXPANSION": {
    #     "func": "range_expansion",
    #     "args": ["High", "Low", "Close"],
    #     "kwargs": {"period": 20}
    # },

    # =====================================================
    # ================= MOMENTUM ACCELERATION ============
    # =====================================================

    "MOMENTUM_ACCEL": {
        "func": "momentum_acceleration",
        "args": ["Close"],
        "kwargs": {"short": 5, "long": 20}
    }
}

        
    