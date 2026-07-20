from dataclasses import dataclass


@dataclass
class Strategy:
    name: str = "up5_low_vol"
    timeframe: str = "1d"

    instruments = [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X",
        "IEF", "TLT", "SHY",
        "GC=F", "SI=F"
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
            "n_estimators": 250,
            "max_depth": 4,
            "learning_rate": 0.05
        }
    }

    features = {

        # mean reversion
        "RSI_14": {"func": "rsi", "args": ["Close"], "kwargs": {"period": 14}},
        "RSI_CENTERED": {"func": "rsi_centered", "args": ["RSI_14"], "kwargs": {}},
        "RSI_REGIME_3": {"func": "rsi_regime_3", "args": ["RSI_14"], "kwargs": {}},

        # range positioning
        "DIST_FROM_HIGH_60": {"func": "dist_from_high", "args": ["Close"], "kwargs": {"period": 60}},
        "DIST_FROM_LOW_60": {"func": "dist_from_low", "args": ["Close"], "kwargs": {"period": 60}},

        # volatility
        "ATR_PCT": {"func": "atr_pct", "args": ["High","Low","Close"], "kwargs": {"period": 14}},
        "ATR_PCT_Z": {"func": "atr_pct_zscore", "args": ["ATR_PCT"], "kwargs": {"period": 60}},
        "ATR_14": {"func": "atr", "args": ["High","Low","Close"], "kwargs": {"period": 14}},

        # volume (slab signal ali ok)
        "VOLUME_Z": {"func": "volume_zscore", "args": ["Volume"], "kwargs": {"period": 20}},
    }