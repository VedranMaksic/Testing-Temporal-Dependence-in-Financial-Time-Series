from dataclasses import dataclass


@dataclass
class Strategy:
    name: str = "up5_mid_vol"
    timeframe: str = "1d"

    instruments = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA",
        "^GSPC", "^NDX", "^DJI", "^RUT"
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
            "max_depth": 5,
            "learning_rate": 0.05
        }
    }

    features = {

        # trend
        "SMA_10": {"func": "sma", "args": ["Close"], "kwargs": {"period": 10}},
        "SMA_50": {"func": "sma", "args": ["Close"], "kwargs": {"period": 50}},
        "EMA_20": {"func": "ema", "args": ["Close"], "kwargs": {"period": 20}},
        "SMA_CROSS_10_50": {"func": "sma_cross", "args": ["SMA_10","SMA_50"], "kwargs": {}},

        "ATR_14": {"func": "atr", "args": ["High","Low","Close"], "kwargs": {"period": 14}},
        "TREND_STRENGTH": {"func": "trend_strength", "args": ["EMA_20","SMA_50","ATR_14"], "kwargs": {}},

        # momentum
        "RSI_14": {"func": "rsi", "args": ["Close"], "kwargs": {"period": 14}},
        "RSI_CENTERED": {"func": "rsi_centered", "args": ["RSI_14"], "kwargs": {}},
        "RSI_CROSS_50": {"func": "rsi_cross_50", "args": ["RSI_14"], "kwargs": {}},

        # structure
        "DIST_FROM_HIGH_60": {"func": "dist_from_high", "args": ["Close"], "kwargs": {"period": 60}},

        # volume
        "OBV": {"func": "obv", "args": ["Close","Volume"], "kwargs": {}},
        "VOLUME_Z": {"func": "volume_zscore", "args": ["Volume"], "kwargs": {"period": 20}},
    }