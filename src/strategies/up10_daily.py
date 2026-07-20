from dataclasses import dataclass


@dataclass
class Strategy:
    name: str = "up10_daily"
    timeframe: str = "1d"

    #time split
    split_date: str = "2023-01-01"

    # Target definicija
    target = {
        "module": "up_down",
        "params": {
            "direction": "up",
            "horizon": 60,
            "threshold": 0.10
        }
    }

    # Model definicija
    model = {
        "module": "xgboost_trainer",
        "params": {
            "n_estimators": 300, #300
            "max_depth": 5, #5
            "learning_rate": 0.05 #0.05
        }
    }

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

    # Feature definicija (feature_name: (function_name, params))
    features = {
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

        "RSI_14": {
            "func": "rsi",
            "args": ["Close"],
            "kwargs": {"period": 14}
        },

        "ATR_14": {
            "func": "atr",
            "args": ["High", "Low", "Close"],
            "kwargs": {"period": 14}
        },

        "ROC_10": {
            "func": "roc",
            "args": ["Close"],
            "kwargs": {"period": 10}
        },

        "OBV": {
            "func": "obv",
            "args": ["Close", "Volume"],
            "kwargs": {}
        }
    }

        
    