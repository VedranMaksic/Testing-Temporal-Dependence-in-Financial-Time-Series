import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, confusion_matrix, r2_score


# =========================
# Paths (hardcoded project layout)
# =========================
PROJECT_ROOT = Path(r"C:\Users\maksi\Desktop\fer\3. godina apsolvent\zavrsni rad\project")
MODELS_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "reports" / "metrics"


# =========================
# Helpers
# =========================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Spearman = Pearson korelacija rangova (rank->Pearson)
    yt = pd.Series(y_true).rank(method="average").to_numpy()
    yp = pd.Series(y_pred).rank(method="average").to_numpy()
    if len(yt) < 3:
        return float("nan")
    return float(np.corrcoef(yt, yp)[0, 1])


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def sanitize_filename(s: str) -> str:
    
    s = str(s).replace("/", "_").replace("\\", "_").replace(":", "_")
    s = s.replace("*", "_").replace("?", "_").replace("\"", "_")
    s = s.replace("<", "_").replace(">", "_").replace("|", "_")
    s = s.replace(" ", "_")
    return s


# =========================
# Classification plots
# =========================
def plot_classification(df_cls: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    required = {"target", "y_true", "y_pred"}
    if not required.issubset(df_cls.columns):
        print("[WARN] classification_predictions.csv missing required columns. Found:", list(df_cls.columns))
        return

    has_proba = "y_proba" in df_cls.columns

    for target, g in df_cls.groupby("target"):
        y_true = g["y_true"].astype(int).to_numpy()
        y_pred = g["y_pred"].astype(int).to_numpy()

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        plt.figure(figsize=(5.6, 4.6))

        
        im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)

        plt.title(f"Confusion Matrix – {target}")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])

        
        thresh = cm.max() / 2.0

        for i in range(2):
            for j in range(2):
                plt.text(
                    j, i,
                    f"{cm[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if cm[i, j] > thresh else "black"
                )

        plt.xlabel("Predicted")
        plt.ylabel("True")

        plt.tight_layout()
        savefig(out_dir / f"{prefix}_cm_{target}.png")


        # ROC curve
        if has_proba and len(np.unique(y_true)) == 2:
            y_proba = g["y_proba"].astype(float).to_numpy()
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(5.6, 4.6))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.title(f"ROC – {target}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            savefig(out_dir / f"{prefix}_roc_{target}.png")


# =========================
# Regression plots
# =========================
def plot_regression(df_reg: pd.DataFrame, out_dir: Path, prefix: str, min_n: int = 50) -> None:
    required = {"target", "y_true", "y_pred"}
    if not required.issubset(df_reg.columns):
        print("[WARN] regression_predictions.csv missing required columns. Found:", list(df_reg.columns))
        return

    for target, g in df_reg.groupby("target"):
        y_true = g["y_true"].astype(float).to_numpy()
        y_pred = g["y_pred"].astype(float).to_numpy()

        # Histogram target
        plt.figure(figsize=(6.3, 4.6))
        plt.hist(y_true, bins=60)
        plt.title(f"Target distribution – {target}")
        plt.xlabel("y_true")
        plt.ylabel("count")
        savefig(out_dir / f"{prefix}_hist_{target}.png")

        
        if str(target).endswith("_log"):
            y_back = np.expm1(y_true)
            plt.figure(figsize=(6.3, 4.6))
            plt.hist(y_back, bins=60)
            plt.title(f"Back-transformed – {target} → raw return")
            plt.xlabel("expm1(y_true)")
            plt.ylabel("count")
            savefig(out_dir / f"{prefix}_hist_back_{target}.png")

        # Scatter y_true vs y_pred (global)
        r2 = float(r2_score(y_true, y_pred))
        sp = float(spearman_corr(y_true, y_pred))

        plt.figure(figsize=(5.8, 5.0))
        plt.scatter(y_true, y_pred, s=8, alpha=0.4)
        mn = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
        mx = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
        plt.plot([mn, mx], [mn, mx], linestyle="--")
        plt.title(f"y_true vs y_pred – {target}\nR²={r2:.3f} | Spearman={sp:.3f}")
        plt.xlabel("y_true")
        plt.ylabel("y_pred")
        savefig(out_dir / f"{prefix}_scatter_{target}.png")

        # Per-instrument plots + tables 
        if "Instrument" in g.columns:
            rows = []
            for inst, gi in g.groupby("Instrument"):
                if len(gi) < min_n:
                    continue
                yt = gi["y_true"].astype(float).to_numpy()
                yp = gi["y_pred"].astype(float).to_numpy()
                rows.append({
                    "Instrument": inst,
                    "n_obs": int(len(gi)),
                    "spearman": spearman_corr(yt, yp),
                    "r2": float(r2_score(yt, yp)),
                })

            if rows:
                res = pd.DataFrame(rows).sort_values("spearman", ascending=False)
                res.to_csv(out_dir / f"{prefix}_per_instrument_{target}.csv", index=False)

                # Bar chart Spearman
                plt.figure(figsize=(9.2, 4.8))
                plt.bar(res["Instrument"], res["spearman"])
                plt.title(f"Spearman by instrument – {target}")
                plt.xlabel("Instrument")
                plt.ylabel("Spearman")
                plt.xticks(rotation=45, ha="right")
                savefig(out_dir / f"{prefix}_bar_spearman_{target}.png")

                # Scatter Spearman vs R²
                plt.figure(figsize=(6.3, 4.8))
                plt.scatter(res["spearman"], res["r2"], s=35, alpha=0.8)
                for _, row in res.iterrows():
                    plt.text(row["spearman"], row["r2"], str(row["Instrument"]), fontsize=8)
                plt.axvline(0, linestyle="--")
                plt.axhline(0, linestyle="--")
                plt.title(f"Spearman vs R² (per instrument) – {target}")
                plt.xlabel("Spearman")
                plt.ylabel("R²")
                savefig(out_dir / f"{prefix}_scatter_spearman_vs_r2_{target}.png")

                
                # Scatter y_true vs y_pred PO INSTRUMENTU
                
                inst_dir = out_dir / "per_instrument_scatter" / sanitize_filename(target)
                ensure_dir(inst_dir)

                for inst, gi in g.groupby("Instrument"):
                    if len(gi) < min_n:
                        continue

                    yt = gi["y_true"].astype(float).to_numpy()
                    yp = gi["y_pred"].astype(float).to_numpy()

                    r2_i = float(r2_score(yt, yp))
                    sp_i = float(spearman_corr(yt, yp))

                    plt.figure(figsize=(5.6, 5.0))
                    plt.scatter(yt, yp, s=10, alpha=0.5)

                    mn = float(np.nanmin([np.nanmin(yt), np.nanmin(yp)]))
                    mx = float(np.nanmax([np.nanmax(yt), np.nanmax(yp)]))
                    plt.plot([mn, mx], [mn, mx], linestyle="--")

                    plt.title(f"{inst} – {target}\nR²={r2_i:.3f} | Spearman={sp_i:.3f}")
                    plt.xlabel("y_true")
                    plt.ylabel("y_pred")

                    fname = f"{prefix}_scatter_{sanitize_filename(target)}_{sanitize_filename(inst)}.png"
                    savefig(inst_dir / fname)


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_set", choices=["base", "enhanced"], required=True)
    ap.add_argument("--min_n", type=int, default=50)
    args = ap.parse_args()

    run_dir = MODELS_DIR / f"ml_output_global_{args.feature_set}"
    cls_path = run_dir / "classification_predictions.csv"
    reg_path = run_dir / "regression_predictions.csv"

    out_dir = OUT_DIR / args.feature_set
    ensure_dir(out_dir)

    print("========================================")
    print("PLOT METRICS")
    print("Feature set:", args.feature_set)
    print("Input dir  :", str(run_dir))
    print("Output dir :", str(out_dir))
    print("========================================")

    if cls_path.exists():
        df_cls = pd.read_csv(cls_path)
        plot_classification(df_cls, out_dir, prefix=args.feature_set)
    else:
        print("[WARN] Missing:", str(cls_path))

    if reg_path.exists():
        df_reg = pd.read_csv(reg_path)
        plot_regression(df_reg, out_dir, prefix=args.feature_set, min_n=args.min_n)
    else:
        print("[WARN] Missing:", str(reg_path))

    print("\n✓ Done. Saved PNGs to:", str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
