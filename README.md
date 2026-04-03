# 🤖 SmartPredictor

A unified Auto-ML library that **automatically detects your problem type** and trains an optimized stacking ensemble — all from a single config file or via an interactive Streamlit UI.

> Built from scratch on scikit-learn, XGBoost, CatBoost, and LightGBM with Bayesian hyperparameter optimization.

---

## ✨ Features

- **Auto problem detection** — binary, multiclass, multilabel, or continuous regression
- **Stacking ensembles** — XGBoost + CatBoost + HistGradientBoosting + LightGBM
- **Bayesian optimization** — `skopt` with configurable iterations
- **Smart preprocessing** — `IterativeImputer`, `RobustScaler`, `OneHotEncoder` via `ColumnTransformer`
- **SMOTE balancing** — automatic class imbalance handling
- **Training history** — stores optimization results and evaluation metrics per run
- **Streamlit UI** — interactive interface with dataset upload, live config, and results visualization
- **CLI mode** — run directly from `config.py`

---

## 🖥️ Streamlit UI

```bash
streamlit run app.py
```

**Workflow :**
1. Upload your dataset (CSV, JSON, PKL, XLS)
2. Select target column and problem type
3. Configure optimization settings in the sidebar
4. Click **Démarrer** — results persist even if you change widgets after

**Displays after training :**
- Models used in the stacking ensemble
- Per-model Bayesian optimization results (best score, best params, duration)
- Evaluation metrics (accuracy, confusion matrix, classification report, etc.)
- Predictions table (in predict/fit_predict mode)

---

## ⚙️ CLI Configuration

Edit `config.py` before running `main.py` :

```python
CONFIG = {
    "dataset_path": "path/to/your/dataset.csv",  # .csv, .json, .pkl, .joblib
    "target_name": "label",                       # Column to predict
    "problem_type": "binary",                     # binary | multiclass | multilabel | continuous
    "what": "fit",                                # fit | predict | fit_predict
    "opt": True,                                  # Enable Bayesian optimization
    "max_iter": 10,                               # Optimization iterations per model
    "model_name": "model.pkl",                    # Used for predict mode
    "smote": False,                               # SMOTE oversampling
    "v": False                                    # Verbose mode
}
```

```bash
python main.py
```

---

## 📁 Project Structure

```
smart-predictor/
├── app.py               # Streamlit UI
├── main.py              # CLI entry point
├── classification.py    # Binary / Multiclass / Multilabel models
├── regression.py        # Stacking regression model
├── config.py            # User configuration
├── dict_model.py        # Maps problem type → model class
├── data_explain.py      # Dataset analysis
├── exception.py         # Custom exceptions
└── __init__.py
```

---

## 🏗️ Model Architectures

### Classification (Binary / Multiclass)

```
StackingClassifier
├── XGBClassifier          (Bayesian optimized)
├── HistGradientBoosting   (Bayesian optimized)
├── RandomForestClassifier (Bayesian optimized)
└── Meta: LogisticRegressionCV + PolynomialFeatures
```

### Multilabel Classification

```
OneVsRestClassifier
└── StackingClassifier (same as above)
```

### Regression

```
StackingRegressor
├── XGBRegressor    (Bayesian optimized)
├── LGBMRegressor   (Bayesian optimized)
├── HistGBRegressor (Bayesian optimized)
├── RandomForest    (Bayesian optimized)
└── Meta: RidgeCV
```

---

## 📊 Training History

After training, access results programmatically :

```python
m = ModelBinaire("data.csv")
m.fit(target_name="label", opt=True)

print(m.history["models"])     # ['xgb', 'hist', 'rf', 'log_reg']
print(m.history["opt"]["xgb"]) # {'best_score_cv': 0.97, 'best_params': {...}, 'duration_sec': 42.3}
print(m.history["eval"])       # {'accuracy': 0.99, 'confusion_matrix': [...], ...}
```

---

## 📦 Installation

```bash
git clone https://github.com/hounsoubenny-cyber/smart-predictor.git
cd smart-predictor
pip install -r requirements.txt
streamlit run app.py
```

**requirements.txt :**
```
scikit-learn>=1.3.0
xgboost>=2.0.0
catboost>=1.2.0
lightgbm>=4.0.0
imbalanced-learn>=0.11.0
scikit-optimize>=0.9.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
streamlit>=1.30.0
```

---

## 🧪 Supported Dataset Formats

| Format    | Notes                          |
|-----------|--------------------------------|
| `.csv`    | Standard comma-separated       |
| `.json`   | List of dicts or dict of lists |
| `.pkl`    | Pickled pandas DataFrame       |
| `.joblib` | Joblib-serialized DataFrame    |
| `.xls`    | Excel format                   |

---

## 👤 Author

**Samuel Hounsou**
- GitHub: [@hounsoubenny-cyber](https://github.com/hounsoubenny-cyber)
- LinkedIn: [benny-hounsou](https://linkedin.com/in/benny-hounsou-00a267374)

---

⭐ Star this repo if it helped you ship ML faster!
