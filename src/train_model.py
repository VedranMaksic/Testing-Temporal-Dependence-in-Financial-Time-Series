import argparse
import importlib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_strategy(strategy_name):
    module = importlib.import_module(f"src.strategies.{strategy_name}")
    return module.Strategy()


def load_features(strategy):
    path = ROOT / "data" / "processed" / f"{strategy.name}_{strategy.timeframe}_features.csv"

    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index(["Date", "Instrument"]).sort_index()

    return df


def load_target_module(strategy):
    module_name = strategy.target["module"]
    return importlib.import_module(f"src.targets.{module_name}")


def load_model_module(strategy):
    module_name = strategy.model["module"]
    return importlib.import_module(f"src.trainers.{module_name}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--mode", choices=["static", "rolling"], default="static")
    parser.add_argument("--window", type=int, default=None,
                        help="Window size in years for rolling window (if None = expanding)")
    args = parser.parse_args()

    strategy = load_strategy(args.strategy)

    print("\nStrategy:", strategy.name)
    print("Timeframe:", strategy.timeframe)
    print("Mode:", args.mode)

    df = load_features(strategy)

    target_module = load_target_module(strategy)

    df = target_module.generate(
        df,
        **strategy.target["params"]
    )

    df = df.dropna(subset=["target"])
    df = df.sort_index()

    feature_cols = list(strategy.features.keys())
    split_date = pd.to_datetime(strategy.split_date)

    # ============================================
    # STATIC MODE
    # ============================================
    if args.mode == "static":

        print("Split date:", split_date)

        dates = df.index.get_level_values("Date")

        train_df = df[dates < split_date]
        test_df = df[dates >= split_date]

        print("Train period:",
              train_df.index.get_level_values("Date").min(),
              "→",
              train_df.index.get_level_values("Date").max())

        print("Test period:",
              test_df.index.get_level_values("Date").min(),
              "→",
              test_df.index.get_level_values("Date").max())

        X_train = train_df[feature_cols]
        y_train = train_df["target"]

        X_test = test_df[feature_cols]
        y_test = test_df["target"]

        model_module = load_model_module(strategy)
        trainer = model_module.Trainer(**strategy.model["params"])

        trainer.train(X_train, y_train)
        y_proba = trainer.predict_proba(X_test)

        results_df = test_df.copy()
        results_df["y_true"] = y_test
        results_df["y_proba"] = y_proba
        results_df["y_pred_0_5"] = (y_proba >= 0.5).astype(int)


        if strategy.model['module'] == "xgboost_trainer":
            importances = trainer.model.feature_importances_
            importance_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": importances
            }).sort_values("Importance", ascending=False)

            print("\n===== FEATURE IMPORTANCE (STATIC) =====")
            print(importance_df.head(25))
    # ============================================
    # ROLLING MODE (FIXED WINDOW + CHUNKS)
    # ============================================
    if args.mode == "rolling":
        print("Rolling start:", split_date)

        all_predictions = []

        dates = df.index.get_level_values("Date").sort_values().unique()

        # podjela
        train_initial = dates[dates < split_date]
        test_all = dates[dates >= split_date]

        n_chunks = 3
        chunk_size = len(test_all) // n_chunks

        # window = 3 chunka 
        window_size = 5 * chunk_size

        print(f"Initial train size: {len(train_initial)}")
        print(f"Chunk size: {chunk_size}")
        print(f"Train window: {window_size}")

        for i in range(n_chunks):

            test_start = i * chunk_size
            test_end = (i + 1) * chunk_size if i < n_chunks - 1 else len(test_all)

            test_dates = test_all[test_start:test_end]

            if len(test_dates) == 0:
                continue

            # TRAIN WINDOW (FIXED)
            train_end_date = test_dates[0]

            train_candidates = dates[dates < train_end_date]

            train_dates = train_candidates[-window_size:]

            if len(train_dates) == 0:
                continue

            train_df = df.loc[
                df.index.get_level_values("Date").isin(train_dates)
            ]

            test_df = df.loc[
                df.index.get_level_values("Date").isin(test_dates)
            ]

            print(f"\nSplit {i+1}")
            print("Train:",
                train_dates[0],
                "→",
                train_dates[-1])

            print("Test:",
                test_dates[0],
                "→",
                test_dates[-1])

            X_train = train_df[feature_cols]
            y_train = train_df["target"]

            X_test = test_df[feature_cols]
            y_test = test_df["target"]

            model_module = load_model_module(strategy)
            trainer = model_module.Trainer(**strategy.model["params"])

            trainer.train(X_train, y_train)
            y_proba = trainer.predict_proba(X_test)

            temp_df = test_df.copy()
            temp_df["y_true"] = y_test
            temp_df["y_proba"] = y_proba
            temp_df["y_pred_0_5"] = (y_proba >= 0.5).astype(int)

            all_predictions.append(temp_df)

            # ✅ FEATURE IMPORTANCE (nije maknut!)
            if strategy.model['module'] == "xgboost_trainer":
                importances = trainer.model.feature_importances_
                importance_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance": importances
                }).sort_values("Importance", ascending=False)

                print("\n===== FEATURE IMPORTANCE =====")
                print(importance_df.head(15))

        if len(all_predictions) == 0:
            print("No predictions generated.")
            return

        results_df = pd.concat(all_predictions).sort_index()

        
    # ============================================
    # SAVE RESULTS
    # ============================================
    out_dir = ROOT / "models" / strategy.name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "predictions.csv"
    results_df.reset_index().to_csv(out_path, index=False)

    print("\nPredictions saved to:", out_path)


if __name__ == "__main__":
    main()