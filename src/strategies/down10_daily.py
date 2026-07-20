from dataclasses import dataclass


@dataclass
class Strategy:
    name: str = "down10_daily"
    timeframe: str = "1d"   

    #time split
    split_date: str = "2023-01-01"

    # Target definicija
    target = {
        "module": "up_down",
        "params": {
            "direction": "down",
            "horizon": 60,
            "threshold": 0.10
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

    
    "ATR_14": { 
        "func": "atr",
        "args": ["High", "Low", "Close"],
        "kwargs": {"period": 14}
    },

    "RSI_14": {
        "func": "rsi",
        "args": ["Close"],
        "kwargs": {"period": 14}
    },
    "OBV": {
        "func": "obv",
        "args": ["Close", "Volume"],
        "kwargs": {}
    },
    
    "ROC_10": {
        "func": "roc",
        "args": ["Close"],
        "kwargs": {"period": 10}
    }

    
}

        
    