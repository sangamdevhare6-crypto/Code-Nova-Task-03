# Sales Prediction Model

> **Synthetic data demo** — all data and results are generated artificially for educational purposes only.

---

## Overview

A complete, beginner-friendly machine-learning project that:

- Generates realistic **synthetic sales data** (≈ 450+ records)
- Cleans and preprocesses the data with a scikit-learn **Pipeline + ColumnTransformer**
- Trains and compares **3 regression algorithms**
- Evaluates models with R², MAE, and RMSE
- Produces **professional visualisation graphs**
- Serves an interactive **Streamlit dashboard** for live predictions

---

## Features

| Feature | Detail |
|---------|--------|
| Algorithms | Linear Regression, Decision Tree, Random Forest |
| Preprocessing | Median / mode imputation, One-Hot Encoding, StandardScaler |
| Selection criterion | Lowest RMSE on held-out test set |
| Split strategy | Chronological 80 / 20 — no data leakage |
| Output artefacts | Trained model (.joblib), graphs (.png), CSVs, summary JSON |

---

## Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| `Date` | string | Sale date (YYYY-MM-DD) |
| `Store` | string | Store identifier (A–D) |
| `Product` | string | Product category |
| `Price` | float | Unit price in $ |
| `Advertising_Spend` | float | Daily ad budget in $ |
| `Promotion` | int (0/1) | Promotion active flag |
| `Holiday` | int (0/1) | Weekend / holiday flag |
| `Sales` | float | **Target** — daily sales in $ |

---

## Installation

> Requires **Python 3.9+** and pip.

```bash
# 1. Clone / unzip the project folder
cd Sales_Prediction_Model

# 2. (Recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows

# 3. Install dependencies
python -m pip install -r requirements.txt
```

---

## Running the Project

### Step 1 — Train the model
```bash
python main.py
```
This will:
- Generate `data/sales_data.csv` if missing
- Clean and preprocess the data
- Train Linear Regression, Decision Tree, and Random Forest
- Print evaluation metrics
- Save model, graphs, and CSVs to the `output/` folder

### Step 2 — Launch the dashboard
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

### One-click (Windows)
```bash
run_project.bat
```

---

## Metric Explanations

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **R²** | 1 − SS_res / SS_tot | Fraction of variance explained; 1.0 = perfect fit |
| **MAE** | mean(|y − ŷ|) | Average absolute error in the same unit as Sales ($) |
| **RMSE** | √mean((y − ŷ)²) | Like MAE but penalises large errors more heavily |

---

## Project Structure

```
Sales_Prediction_Model/
├── data/
│   └── sales_data.csv          ← synthetic dataset
├── output/
│   ├── best_sales_model.joblib ← trained pipeline
│   ├── cleaned_sales_data.csv
│   ├── model_comparison.csv
│   ├── predictions.csv
│   ├── summary.json
│   └── graphs/
│       ├── actual_vs_predicted.png
│       ├── model_comparison.png
│       ├── sales_trend.png
│       └── feature_importance.png
├── main.py                     ← training pipeline
├── app.py                      ← Streamlit dashboard
├── requirements.txt
├── run_project.bat
└── README.md
```

---

## Limitations

- **Synthetic data only** — results do not reflect real business outcomes
- The Random Forest model may overfit to synthetic noise patterns
- No hyperparameter tuning (GridSearchCV) is applied in this demo
- Predictions are estimates; always validate with domain knowledge

---

## License

This project is provided as an open educational resource. No real company data is used.
