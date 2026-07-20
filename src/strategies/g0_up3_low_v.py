from dataclasses import dataclass


@dataclass
class Strategy:
    name: str = "g0_up3_low_v"
    timeframe: str = "1d"

    instruments = [
    

    # =========================
    # METALS
    # =========================
    "GC=F",
   
    

    # =========================
    # STOCKS (TECH / BIG CAP)
    # =========================
    
    
    "NVDA",
    

    # =========================
    # INDICES
    # =========================
    "^GSPC",
    "^NDX",
    

 
    #index
    "S&P 500"
    ]

    #time split
    split_date: str = "2023-01-01"

    # Target definicija
    target = {
        "module": "up_down",
        "params": {
            "direction": "up",
            "horizon": 60,
            "threshold": 0.03
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

   
    "ATR_14": { #added here because trend strength is next
        "func": "atr",
        "args": ["High", "Low", "Close"],
        "kwargs": {"period": 14}
    },

    "ROC_10": {
            "func": "roc",
            "args": ["Close"],
            "kwargs": {"period": 10}
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

        
    