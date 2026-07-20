from dataclasses import dataclass


@dataclass
class Strategy:
    name: str = "up5_high_vol"
    timeframe: str = "1d"

    instruments = [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD",
        "XRP-USD", "ADA-USD", "DOGE-USD",
        "TSLA", "NVDA"
    ]

    split_date: str = "2023-01-01"

    target = {
        "module": "up_down",
        "params": {
            "direction": "up",
            "horizon": 60,
            "threshold": 0.05
        }
    }

    model = {
        "module": "xgboost_trainer",
        "params": {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05
        }
    }

    features = {

        # momentum
        "RSI_14": {"func": "rsi", "args": ["Close"], "kwargs": {"period": 14}},
        "RSI_CROSS_50": {"func": "rsi_cross_50", "args": ["RSI_14"], "kwargs": {}},

        "MOMENTUM_ACCEL": {
            "func": "momentum_acceleration",
            "args": ["Close"],
            "kwargs": {"short": 5, "long": 20}
        },

        # volatility
        "LOG_RETURN_1": {"func": "log_return", "args": ["Close"], "kwargs": {"period": 1}},
        "ROLLING_VOL_20": {"func": "rolling_volatility", "args": ["LOG_RETURN_1"], "kwargs": {"period": 20}},
        "ATR_PCT": {"func": "atr_pct", "args": ["High","Low","Close"], "kwargs": {"period": 14}},
        "ATR_PCT_Z": {"func": "atr_pct_zscore", "args": ["ATR_PCT"], "kwargs": {"period": 60}},
        "ATR_14": {"func": "atr", "args": ["High","Low","Close"], "kwargs": {"period": 14}},

        # breakout
        "DIST_FROM_HIGH_60": {"func": "dist_from_high", "args": ["Close"], "kwargs": {"period": 60}},
        "RANGE_EXPANSION": {"func": "range_expansion", "args": ["High","Low","Close"], "kwargs": {"period": 20}},

        # volume
        "OBV": {"func": "obv", "args": ["Close","Volume"], "kwargs": {}},
        "OBV_SLOPE_5": {"func": "obv_slope_n", "args": ["OBV"], "kwargs": {"period": 5}},
    }