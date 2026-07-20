from dataclasses import dataclass


@dataclass
class Strategy:
    name: str = "up1_aapl_hourly"
    timeframe: str = "1h"

    instruments = [
    
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",

    
    ]

    #time split
    split_date: str = "2025-10-10"

    # Target definicija
    target = {
        "module": "up_down",
        "params": {
            "direction": "up",
            "horizon": 60,
            "threshold": 0.005
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
    "ATR_14": {
        "func": "atr",
        "args": ["High", "Low", "Close"],
        "kwargs": {"period": 14}
    },

    "TREND_STRENGTH": {
        "func": "trend_strength",
        "args": ["EMA_20", "SMA_50", "ATR_14"],
        "kwargs": {}
    },

    

    "RSI_14": {
        "func": "rsi",
        "args": ["Close"],
        "kwargs": {"period": 14}
    },

    "ROC_10": {
            "func": "roc",
            "args": ["Close"],
            "kwargs": {"period": 10}
    },

   
    "ATR_PCT": {
        "func": "atr_pct",
        "args": ["High", "Low", "Close"],
        "kwargs": {"period": 14}
    },

    

    "OBV": {
        "func": "obv",
        "args": ["Close", "Volume"],
        "kwargs": {}
    }


}

        
    