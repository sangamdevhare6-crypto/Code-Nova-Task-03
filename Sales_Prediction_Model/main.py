"""
Sales Prediction Model — Training Pipeline
==========================================
Generates synthetic data (if missing), preprocesses it, trains three
regression models, evaluates them, saves the best model, and produces
all required visualisation graphs.

Run:
    python main.py
"""

import os
import sys
import json
import warnings
import random
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
DATA_PATH      = os.path.join("data", "sales_data.csv")
OUTPUT_DIR     = "output"
GRAPHS_DIR     = os.path.join(OUTPUT_DIR, "graphs")
MODEL_PATH     = os.path.join(OUTPUT_DIR, "best_sales_model.joblib")
COMPARISON_CSV = os.path.join(OUTPUT_DIR, "model_comparison.csv")
PREDICTIONS_CSV= os.path.join(OUTPUT_DIR, "predictions.csv")
CLEANED_CSV    = os.path.join(OUTPUT_DIR, "cleaned_sales_data.csv")
SUMMARY_JSON   = os.path.join(OUTPUT_DIR, "summary.json")

RANDOM_SEED = 42
TARGET_COL  = "Sales"

os.makedirs(GRAPHS_DIR, exist_ok=True)

# ── 0. Synthetic Dataset Generator ──────────────────────────────────────────
def generate_dataset(path: str, n_records_target: int = 450) -> None:
    """Create a realistic synthetic sales CSV if it does not already exist."""
    log.info("Generating synthetic dataset → %s", path)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    stores   = ["Store_A", "Store_B", "Store_C", "Store_D"]
    products = ["Electronics", "Clothing", "Groceries", "Furniture", "Sports"]
    dates    = pd.date_range("2022-01-01", "2023-12-31", freq="D")

    rows = []
    for date in dates:
        for store in random.sample(stores, k=random.randint(2, 4)):
            product  = random.choice(products)
            price    = round(random.uniform(10, 500), 2)
            ad_spend = round(random.uniform(50, 2000), 2)
            promo    = random.choice([0, 1])
            holiday  = 1 if date.weekday() >= 5 else 0
            noise    = random.gauss(0, 60)
            sales    = max(0, round(
                price * 0.9 + ad_spend * 0.18 + promo * 220
                + holiday * 140 + noise, 2))
            rows.append([date.strftime("%Y-%m-%d"), store, product,
                         price, ad_spend, promo, holiday, sales])

    df = pd.DataFrame(rows, columns=[
        "Date", "Store", "Product", "Price",
        "Advertising_Spend", "Promotion", "Holiday", "Sales"
    ])

    # Inject ~2 % missing values in selected columns
    for col in ["Price", "Advertising_Spend", "Store", "Product"]:
        miss_idx = df.sample(frac=0.02, random_state=RANDOM_SEED).index
        df.loc[miss_idx, col] = np.nan

    # Inject ~1 % duplicate rows
    dups = df.sample(frac=0.01, random_state=7)
    df   = pd.concat([df, dups], ignore_index=True)

    # Shuffle so ordering looks realistic
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    log.info("  → %d rows written (synthetic sample data only)", len(df))


# ── 1. Load & Inspect ────────────────────────────────────────────────────────
def load_and_inspect(path: str) -> pd.DataFrame:
    log.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    log.info("  Shape        : %s", df.shape)
    log.info("  Columns      : %s", list(df.columns))
    log.info("  Missing vals :\n%s", df.isnull().sum().to_string())
    log.info("  Duplicates   : %d", df.duplicated().sum())
    return df


# ── 2. Preprocess ────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Preprocessing …")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    log.info("  Dropped %d duplicates", before - len(df))

    # Parse dates; drop rows with completely invalid dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    bad_dates  = df["Date"].isna().sum()
    if bad_dates:
        log.warning("  Dropping %d rows with unparseable dates", bad_dates)
        df = df.dropna(subset=["Date"])

    # Drop rows where Sales (target) is missing
    miss_target = df[TARGET_COL].isna().sum()
    if miss_target:
        log.warning("  Dropping %d rows with missing Sales", miss_target)
        df = df.dropna(subset=[TARGET_COL])

    # Extract date features
    df["Year"]       = df["Date"].dt.year
    df["Month"]      = df["Date"].dt.month
    df["DayOfWeek"]  = df["Date"].dt.dayofweek
    df["DayOfYear"]  = df["Date"].dt.dayofyear

    df = df.sort_values("Date").reset_index(drop=True)
    log.info("  Final shape  : %s", df.shape)
    return df


# ── 3. Feature Engineering & Split ──────────────────────────────────────────
def build_features_and_split(df: pd.DataFrame):
    """Return X_train, X_test, y_train, y_test + feature list."""
    feature_cols = [
        "Store", "Product",               # categorical
        "Price", "Advertising_Spend",     # numeric (may have NaN)
        "Promotion", "Holiday",            # numeric flags
        "Year", "Month", "DayOfWeek", "DayOfYear"
    ]

    X = df[feature_cols].copy()
    y = df[TARGET_COL].values

    # Chronological 80/20 split — no shuffling to avoid data leakage
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx],       y[split_idx:]
    log.info("  Train size: %d  |  Test size: %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test, feature_cols


# ── 4. Build Scikit-learn Pipeline ──────────────────────────────────────────
def build_pipeline(model) -> Pipeline:
    """
    ColumnTransformer:
      • Numeric  → median imputation → standard scaling
      • Categorical → most-frequent imputation → one-hot encoding
    """
    numeric_features     = ["Price", "Advertising_Spend",
                             "Promotion", "Holiday",
                             "Year", "Month", "DayOfWeek", "DayOfYear"]
    categorical_features = ["Store", "Product"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),  # sklearn >= 1.2
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model",        model),
    ])


# ── 5. Train & Evaluate ──────────────────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train three models, return results dict and predictions of best model."""
    candidates = {
        "Linear Regression":       LinearRegression(),
        "Decision Tree":           DecisionTreeRegressor(random_state=RANDOM_SEED,
                                                         max_depth=8),
        "Random Forest":           RandomForestRegressor(n_estimators=150,
                                                         random_state=RANDOM_SEED,
                                                         n_jobs=-1),
    }

    results  = {}
    pipelines = {}

    for name, estimator in candidates.items():
        log.info("Training: %s …", name)
        pipe = build_pipeline(estimator)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        r2   = r2_score(y_test, preds)
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        results[name]   = {"R2": r2, "MAE": mae, "RMSE": rmse}
        pipelines[name] = pipe
        log.info("  R²=%.4f  MAE=%.2f  RMSE=%.2f", r2, mae, rmse)

    # Best model = lowest RMSE on test set
    best_name = min(results, key=lambda k: results[k]["RMSE"])
    log.info("Best model (lowest RMSE): %s", best_name)
    return results, pipelines, best_name


# ── 6. Feature Importance (tree-based) ──────────────────────────────────────
def get_feature_names(pipeline: Pipeline) -> list:
    """Recover feature names after ColumnTransformer."""
    ct = pipeline.named_steps["preprocessor"]
    try:
        return ct.get_feature_names_out().tolist()
    except Exception:
        return []


def get_feature_importance(pipeline: Pipeline) -> pd.Series | None:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    names = get_feature_names(pipeline)
    imp   = pd.Series(model.feature_importances_, index=names)
    return imp.sort_values(ascending=False)


# ── 7. Visualisations ────────────────────────────────────────────────────────
# Use a version-safe matplotlib style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
    STYLE = "seaborn-v0_8-whitegrid"
except OSError:
    STYLE = "ggplot"

def _save(fig, filename: str) -> None:
    path = os.path.join(GRAPHS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved graph: %s", path)


def plot_actual_vs_predicted(y_test, y_pred, model_name: str) -> None:
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.45, color="#4A90D9",
                   edgecolors="none", s=20, label="Predictions")
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
        ax.set_title(f"Actual vs Predicted Sales — {model_name}", fontsize=14, pad=12)
        ax.set_xlabel("Actual Sales ($)", fontsize=11)
        ax.set_ylabel("Predicted Sales ($)", fontsize=11)
        ax.legend(fontsize=10)
        fig.tight_layout()
    _save(fig, "actual_vs_predicted.png")


def plot_model_comparison(results: dict) -> None:
    metrics = ["R2", "MAE", "RMSE"]
    df_res  = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        colors = ["#4A90D9", "#E87040", "#57C47B"]

        for ax, metric, color in zip(axes, metrics, colors):
            bars = ax.bar(df_res["Model"], df_res[metric], color=color,
                          edgecolor="white", linewidth=0.8)
            ax.set_title(metric, fontsize=13, pad=8)
            ax.set_ylabel(metric, fontsize=10)
            ax.set_xticklabels(df_res["Model"], rotation=15, ha="right", fontsize=9)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8)

        fig.suptitle("Model Comparison — Evaluation Metrics", fontsize=15, y=1.02)
        fig.tight_layout()
    _save(fig, "model_comparison.png")


def plot_sales_trend(df: pd.DataFrame) -> None:
    # 'ME' = month-end in pandas >= 2.2; fall back to 'M' for older versions
    try:
        monthly = df["Sales"].resample("ME").sum().reset_index()
    except ValueError:
        monthly = df["Sales"].resample("M").sum().reset_index()
    monthly.columns = ["Date", "Sales"]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.fill_between(monthly["Date"], monthly["Sales"],
                        alpha=0.25, color="#4A90D9")
        ax.plot(monthly["Date"], monthly["Sales"],
                color="#4A90D9", linewidth=2, marker="o", markersize=4)
        ax.set_title("Monthly Sales Trend", fontsize=14, pad=12)
        ax.set_xlabel("Month", fontsize=11)
        ax.set_ylabel("Total Sales ($)", fontsize=11)
        fig.autofmt_xdate()
        fig.tight_layout()
    _save(fig, "sales_trend.png")


def plot_feature_importance(importance: pd.Series, top_n: int = 20) -> None:
    top = importance.head(top_n)
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, max(4, len(top) * 0.38)))
        ax.barh(top.index[::-1], top.values[::-1],
                color="#57C47B", edgecolor="white", linewidth=0.7)
        ax.set_title(f"Feature Importances (Top {top_n})", fontsize=14, pad=12)
        ax.set_xlabel("Importance", fontsize=11)
        fig.tight_layout()
    _save(fig, "feature_importance.png")


# ── 8. Save Outputs ──────────────────────────────────────────────────────────
def save_outputs(results: dict, best_name: str,
                 best_pipeline: Pipeline,
                 X_test: pd.DataFrame,
                 y_test: np.ndarray,
                 df_clean: pd.DataFrame) -> None:

    # Predictions
    y_pred = best_pipeline.predict(X_test)
    pred_df = X_test.copy()
    pred_df["Actual_Sales"]    = y_test
    pred_df["Predicted_Sales"] = y_pred.round(2)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    log.info("Predictions saved → %s", PREDICTIONS_CSV)

    # Cleaned data
    df_clean.to_csv(CLEANED_CSV, index=False)

    # Model comparison
    comp_df = pd.DataFrame(results).T.rename_axis("Model").reset_index()
    comp_df.to_csv(COMPARISON_CSV, index=False)
    log.info("Comparison CSV saved → %s", COMPARISON_CSV)

    # Best model
    joblib.dump(best_pipeline, MODEL_PATH)
    log.info("Model saved → %s", MODEL_PATH)

    # Summary JSON
    best = results[best_name]
    summary = {
        "best_model":       best_name,
        "selection_criterion": "lowest RMSE on held-out test set",
        "metrics": {
            "R2":   round(best["R2"],   4),
            "MAE":  round(best["MAE"],  2),
            "RMSE": round(best["RMSE"], 2),
        },
        "all_results": {k: {m: round(v, 4) for m, v in vv.items()}
                        for k, vv in results.items()},
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary JSON saved → %s", SUMMARY_JSON)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("Sales Prediction Model — Training Pipeline")
    log.info("=" * 60)

    # 0. Ensure dataset exists
    if not os.path.exists(DATA_PATH):
        generate_dataset(DATA_PATH)

    # 1. Load
    df_raw = load_and_inspect(DATA_PATH)

    # 2. Preprocess
    df_clean = preprocess(df_raw.copy())

    # 3. Split
    X_train, X_test, y_train, y_test, feat_cols = build_features_and_split(df_clean)

    # 4. Train & evaluate
    results, pipelines, best_name = train_and_evaluate(X_train, X_test, y_train, y_test)

    best_pipeline = pipelines[best_name]
    y_pred        = best_pipeline.predict(X_test)

    # 5. Visualisations
    log.info("Generating visualisations …")
    plot_actual_vs_predicted(y_test, y_pred, best_name)
    plot_model_comparison(results)
    plot_sales_trend(df_clean.set_index("Date"))
    importance = get_feature_importance(best_pipeline)
    if importance is not None:
        plot_feature_importance(importance)
    else:
        log.info("  (Feature importance not available for %s)", best_name)

    # 6. Save artefacts
    save_outputs(results, best_name, best_pipeline, X_test, y_test, df_clean)

    log.info("=" * 60)
    log.info("Training complete!")
    log.info("Best model : %s", best_name)
    log.info("R²=%.4f  MAE=%.2f  RMSE=%.2f",
             results[best_name]["R2"],
             results[best_name]["MAE"],
             results[best_name]["RMSE"])
    log.info("Output files are in: %s/", OUTPUT_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
