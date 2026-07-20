from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # ==============================
    # BASIC EXPERIMENT SETTINGS
    # ==============================

    # Feature registry name (ne više Literal!)
    feature_set: str = "enhanced"

    # Strategy type
    # "ml_fixed", "ml_dynamic", "candlestick", "rule_based", itd.
    strategy_type: str = "ml_fixed"

    # Timeframe
    # "1d", "1h"
    timeframe: str = "1d"

    # ==============================
    # TARGET SETTINGS
    # ==============================

    # Za daily horizon u danima
    horizon_days: int = 60

    # Ako koristiš hourly, horizon_steps se može koristiti
    horizon_steps: Optional[int] = None

    up_thresholds: Tuple[float, ...] = (0.10,)
    down_thresholds: Tuple[float, ...] = (0.10,)

    # ==============================
    # STRATEGY PARAMS
    # ==============================

    # Fixed threshold
    base_threshold: float = 0.6

    # Dynamic threshold param (npr volatility scaling)
    dynamic_alpha: float = 0.05

    # ==============================
    # TIME SPLIT
    # ==============================

    train_start: str = "2010-01-01"
    train_end: str = "2022-12-30"
    test_start: str = "2023-01-03"

    # ==============================
    # DATA QUALITY FILTERS
    # ==============================

    min_rows_after_dropna: int = 1500
    min_pos_in_train: int = 50
    min_pos_in_test: int = 20

    # ==============================
    # AUTO-GENERATED PATHS
    # ==============================

    input_csv: str = field(init=False)
    out_dir: str = field(init=False)

    # ==============================
    # INIT
    # ==============================

    def __post_init__(self):

        # processed data ovisi o timeframe-u i feature setu
        self.input_csv = str(
            ROOT
            / "data"
            / "processed"
            / f"all_instruments_features_{self.feature_set}_{self.timeframe}.csv"
        )

        self.out_dir = str(
            ROOT
            / "models"
            / f"ml_output_{self.feature_set}_{self.timeframe}"
        )

    # ==============================
    # FEATURE ACCESS (registry)
    # ==============================

    @property
    def feature_cols(self) -> Tuple[str, ...]:
        from src.indicators.feature_registry import FEATURE_SETS

        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature_set: {self.feature_set}")

        return FEATURE_SETS[self.feature_set]