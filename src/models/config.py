from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Literal

ROOT = Path(__file__).resolve().parents[2]

FeatureSet = Literal["base", "enhanced"]

@dataclass
class Config:
    # odabir feature seta
    feature_set: FeatureSet = "enhanced"   # "base" ili "enhanced"

    # input/output (automatski iz feature_set)
    input_csv: str = field(init=False)
    out_dir: str = field(init=False)

    # targets
    horizon_days: int = 60
    up_thresholds: Tuple[float, ...] = (0.10,)
    down_thresholds: Tuple[float, ...] = (0.10,)

    # time split
    train_start: str = "2010-01-01"
    train_end: str = "2022-12-30"
    test_start: str = "2023-01-03"

    # ---------- BASE FEATURES ----------
    feature_cols_base: Tuple[str, ...] = (
        "SMA_10",
        "SMA_50",
        "EMA_20",
        "RSI_14",
        "ROC_10",
        "ATR_14",
        "OBV",
    )

    # ---------- ENHANCED (CORE, SMISLENI) ----------
    feature_cols_enhanced: Tuple[str, ...] = (
        # base
        "SMA_10",
        "SMA_50",
        "EMA_20",
        "RSI_14",
        "ROC_10",
        "ATR_14",
        "OBV",

        # MA / trend usage
        "SMA10_gt_SMA50",
        "SMA10_cross_SMA50",
        "SPREAD_SMA10_SMA50",
        "DIST_Close_SMA50_rel",
        "DIST_Close_EMA20_rel",
        "SPREAD_EMA20_SMA50",

        # RSI usage
        "RSI_overbought",
        "RSI_oversold",
        "RSI_gt_50",
        "RSI_cross_50",
        "RSI_centered",

        # OBV usage
        "OBV_slope",
        "OBV_div_bear",
        "OBV_div_bull",

        # ATR context
        "ATR_pct",

        #novi u paketima
        "log_close", "z_close_60", "dist_high_60", "dist_low_60",
        "ret_1d", "ret_5d", "ret_20d", "vol_20", "vol_60", "vol_ratio_20_60",
        "atr_pct_z_60", "mom_20_minus_60", "trend_strength",
        "RSI_abs_dist_50", "RSI_regime_3", "RSI_ema_10",
        "vol_z_20", "vol_ratio_5_20", "obv_slope_1", "obv_slope_5", "obv_slope_1_over_atr",
        "Close_above_SMA50", "Close_above_EMA20", "slope_SMA50_10"

    )

    
    min_rows_after_dropna: int = 1500
    min_pos_in_train: int = 50
    min_pos_in_test: int = 20

    def __post_init__(self):
        self.input_csv = str(
            ROOT / "data" / "processed" / f"all_instruments_features_{self.feature_set}.csv"
        )
        self.out_dir = str(
            ROOT / "models" / f"ml_output_global_{self.feature_set}"
        )

    @property
    def feature_cols(self) -> Tuple[str, ...]:
        if self.feature_set == "base":
            return self.feature_cols_base
        if self.feature_set == "enhanced":
            return self.feature_cols_enhanced
        raise ValueError(f"Unknown feature_set: {self.feature_set}")
