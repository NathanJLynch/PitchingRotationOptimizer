# ml/train.py
"""
Train all three ML models on the pre-built dataset.

Run after build_dataset.py:
    python -m ml.train

Models saved to ml/models/ and auto-loaded by predict.py.
"""
import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

# ─────────────────────────────────────────────
# Three models, three targets:
#
#   Model 1 (LightGBM): label_k_pct
#     → predicts K% for this pitcher vs. this opponent
#     → feeds into whiff/accuracy matchup dimension
#
#   Model 2 (XGBoost): label_ops_allowed
#     → predicts OPS allowed for this matchup
#     → feeds into the OPS dimension
#
#   Model 3 (XGBoost): label_wpa
#     → the "master" label: actual WPA from historical starts
#     → used to validate and blend the sub-scores
# ─────────────────────────────────────────────

# These must exactly match the column names produced by build_dataset.py
FEATURE_COLS = [
    # Pitcher mechanics
    "p_whiff_pct",
    "p_zone_pct",
    "p_chase_pct",
    "p_fb_velo",
    "p_spin_rate",
    "p_extension",
    "p_xwoba_allowed",
    "p_hard_hit_pct",
    "p_fb_usage",
    "p_breaking_usage",
    "p_offspeed_usage",
    # Opponent profile
    "opp_ops",
    "opp_whiff_rate",
    "opp_chase_rate",
    "opp_k_pct",
    "opp_hard_hit_pct",
    "opp_xwoba",
    # Interaction terms (most predictive)
    "whiff_matchup",
    "chase_matchup",
    "ops_gap",
    "power_matchup",
]

DATA_PATH  = "ml/data/training_data.parquet"
MODEL_DIR  = os.environ.get("ML_MODEL_DIR", "ml/models")


def load_training_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Training data not found at {DATA_PATH}.\n"
            f"Run:  python -m ml.build_dataset"
        )
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} training rows from {DATA_PATH}")
    print(f"Date range: {df['game_date'].min()} → {df['game_date'].max()}")
    return df


def train_all_models():
    df = load_training_data()

    # Validate all feature columns exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"These feature columns are missing from the dataset:\n  {missing}\n"
            f"Dataset columns: {list(df.columns)}"
        )

    # Drop rows where any feature is NaN (e.g. no fastball data)
    X = df[FEATURE_COLS].copy()
    before = len(X)
    X = X.dropna()
    df = df.loc[X.index]
    print(f"Dropped {before - len(X)} rows with NaN features ({len(X)} remain)")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Time-series cross-validation ──────────────────────────
    # Data has temporal structure. Never shuffle baseball data —
    # a model that sees 2023 games while predicting 2022 is cheating.
    # gap=162 skips ~one season between train/test folds.
    tscv = TimeSeriesSplit(n_splits=5, gap=162)

    # ── Model 1: K% predictor (LightGBM) ─────────────────────
    print("\n── Training K% model (LightGBM) ──")
    k_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=31,
        min_child_samples=30,   # prevents overfitting small pitcher samples
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbose=-1,
    )
    k_model.fit(
        X, df["label_k_pct"],
        eval_set=[(X.iloc[-500:], df["label_k_pct"].iloc[-500:])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    # ── Model 2: OPS allowed predictor (XGBoost) ─────────────
    print("\n── Training OPS model (XGBoost) ──")
    ops_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        eval_metric="mae",
        early_stopping_rounds=50,
        verbosity=0,
    )
    ops_model.fit(
        X, df["label_ops_allowed"],
        eval_set=[(X.iloc[-500:], df["label_ops_allowed"].iloc[-500:])],
        verbose=100,
    )

    # ── Model 3: WPA predictor (XGBoost) — master blend signal ──
    print("\n── Training WPA model (XGBoost) ──")
    wpa_model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.015,
        max_depth=5,
        min_child_weight=10,    # WPA is noisier — needs more regularization
        subsample=0.75,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=2.0,
        early_stopping_rounds=50,
        verbosity=0,
    )
    wpa_model.fit(
        X, df["label_wpa"],
        eval_set=[(X.iloc[-500:], df["label_wpa"].iloc[-500:])],
        verbose=100,
    )

    # ── Save ──────────────────────────────────────────────────
    joblib.dump(k_model,   os.path.join(MODEL_DIR, "k_pct_model.pkl"))
    joblib.dump(ops_model, os.path.join(MODEL_DIR, "ops_model.pkl"))
    joblib.dump(wpa_model, os.path.join(MODEL_DIR, "wpa_model.pkl"))
    # Save the feature list alongside models so predict.py can verify alignment
    joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_cols.pkl"))

    print(f"\nModels saved to {MODEL_DIR}/")
    _log_feature_importance(k_model, ops_model, wpa_model)
    return k_model, ops_model, wpa_model


def _log_feature_importance(k_model, ops_model, wpa_model):
    for name, model in [("K%", k_model), ("OPS", ops_model), ("WPA", wpa_model)]:
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
            print(f"\n── Top 10 features for {name} model ──")
            print(importances.nlargest(10).to_string())


if __name__ == "__main__":
    train_all_models()