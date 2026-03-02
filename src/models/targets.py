import pandas as pd
from src.models.config import Config
import numpy as np

def compute_future_targets(df_inst: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = df_inst.sort_index().copy()

    s = out["Close"].shift(-1)  # Close_{t+1} sjedi na indeksu t
    future_max = s.rolling(horizon, min_periods=horizon).max().shift(-(horizon - 1))
    future_min = s.rolling(horizon, min_periods=horizon).min().shift(-(horizon - 1))

    out[f"MaxRet{horizon}"] = (future_max / out["Close"]) - 1.0
    out[f"MinRet{horizon}"] = (future_min / out["Close"]) - 1.0

     # log target 
    #out[f"MaxRet{horizon}_log"] = np.log1p(out[f"MaxRet{horizon}"])
    #out[f"MinRet{horizon}_log"] = np.log1p(-out[f"MinRet{horizon}"])
    return out



def add_classification_targets(df_inst: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    
    h = cfg.horizon_days

    for thr in cfg.up_thresholds:
        df_inst[f"Up{int(thr*100)}"] = (df_inst[f"MaxRet{h}"] >= thr).astype(int)

    for thr in cfg.down_thresholds:
        df_inst[f"Down{int(thr*100)}"] = (df_inst[f"MinRet{h}"] <= -thr).astype(int)

    return df_inst


def get_cls_targets(cfg: Config):
    return [f"Up{int(t*100)}" for t in cfg.up_thresholds] + [f"Down{int(t*100)}" for t in cfg.down_thresholds]

#privremeno promjena za log
def get_reg_targets(cfg: Config):
    h = cfg.horizon_days
    return [f"MaxRet{h}", f"MinRet{h}"] #staro
    #return [f"MaxRet{h}_log", f"MinRet{h}_log"] #novo

