import argparse
from pathlib import Path
import joblib


def inspect_model(model_path: Path) -> None:
    print("=" * 70)
    print("MODEL FILE:", model_path)
    print("=" * 70)

    model = joblib.load(model_path)

    print("\nPipeline steps:")
    for name, step in model.named_steps.items():
        print(f" - {name}: {type(step)}")

    
    prep = model.named_steps["prep"]
    feature_names = prep.get_feature_names_out()

    print("\nFEATURES USED BY MODEL:")
    for f in feature_names:
        print(" -", f)

    print("\nTOTAL FEATURES:", len(feature_names))


    for key in ["clf", "reg"]:
        if key in model.named_steps:
            booster = model.named_steps[key]
            if hasattr(booster, "feature_importances_"):
                print("\nTOP 20 FEATURE IMPORTANCE:")
                importances = booster.feature_importances_
                pairs = sorted(
                    zip(feature_names, importances),
                    key=lambda x: x[1],
                    reverse=True,
                )
                for name, val in pairs[:20]:
                    print(f" {name:35s} {val:.4f}")
            break


def main():
    parser = argparse.ArgumentParser(description="Inspect saved ML model")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to .joblib model file",
    )
    args = parser.parse_args()

    inspect_model(Path(args.model_path))


if __name__ == "__main__":
    main()
