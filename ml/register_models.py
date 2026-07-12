# ml/register_models.py
"""
Registers trained .pkl models in the MLModelVersion table so
/health/models returns 200 and the audit trail is complete.

Run after ml/train.py:
    python -m ml.register_models
"""
import os
import joblib
import numpy as np
from datetime import date, datetime
import pandas as pd

from database.models import MLModelVersion, get_engine, get_session_factory, get_db_session

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///rotation_optimizer.db")
MODEL_DIR    = os.environ.get("ML_MODEL_DIR", "ml/models")
DATA_PATH    = "ml/data/training_data.parquet"

MODEL_DEFS = [
    {"name": "k_pct_model",  "filename": "k_pct_model.pkl",  "label": "label_k_pct"},
    {"name": "ops_model",    "filename": "ops_model.pkl",     "label": "label_ops_allowed"},
    {"name": "wpa_model",    "filename": "wpa_model.pkl",     "label": "label_wpa"},
]


def compute_val_mae(model, df: pd.DataFrame, label_col: str, feature_cols: list) -> float:
    """Compute MAE on the last 500 rows (the held-out eval set used during training)."""
    X_val = df[feature_cols].iloc[-500:].dropna()
    y_val = df[label_col].iloc[-500:].loc[X_val.index]
    preds = model.predict(X_val)
    return float(np.mean(np.abs(preds - y_val.values)))


def register():
    engine         = get_engine(DATABASE_URL)
    session_factory = get_session_factory(engine)

    # Load training data to compute val MAE and get date range
    if not os.path.exists(DATA_PATH):
        print(f"Training data not found at {DATA_PATH} — val MAE will be skipped.")
        df = None
    else:
        df = pd.read_parquet(DATA_PATH)
        print(f"Loaded {len(df)} rows from {DATA_PATH}")

    # Load feature cols saved alongside models
    feature_cols_path = os.path.join(MODEL_DIR, "feature_cols.pkl")
    if os.path.exists(feature_cols_path):
        feature_cols = joblib.load(feature_cols_path)
    else:
        # Fallback: import directly from train.py
        from ml.train import FEATURE_COLS
        feature_cols = FEATURE_COLS

    training_cutoff = (
        df["game_date"].max().date() if df is not None else date.today()
    )
    version = date.today().isoformat()   # e.g. "2025-06-17"

    with get_db_session(session_factory) as db:
        for m in MODEL_DEFS:
            pkl_path = os.path.join(MODEL_DIR, m["filename"])
            if not os.path.exists(pkl_path):
                print(f"  ✗ {m['name']}: {pkl_path} not found — skipping")
                continue

            model    = joblib.load(pkl_path)
            val_mae  = compute_val_mae(model, df, m["label"], feature_cols) if df is not None else None

            # Deactivate any previously active version of this model
            db.query(MLModelVersion).filter_by(
                model_name=m["name"], is_active=True
            ).update({"is_active": False})

            # Check if this exact version already exists (idempotent re-runs)
            existing = db.query(MLModelVersion).filter_by(
                model_name=m["name"], version=version
            ).first()

            if existing:
                existing.is_active       = True
                existing.val_mae         = val_mae
                existing.artifact_path   = pkl_path
                existing.training_cutoff = training_cutoff
                existing.feature_list    = feature_cols
                print(f"  ↺ {m['name']} v{version} updated (val_mae={val_mae:.4f})")
            else:
                db.add(MLModelVersion(
                    model_name       = m["name"],
                    version          = version,
                    artifact_path    = pkl_path,
                    training_cutoff  = training_cutoff,
                    val_mae          = val_mae,
                    feature_list     = feature_cols,
                    is_active        = True,
                ))
                print(f"  ✓ {m['name']} v{version} registered (val_mae={val_mae:.4f})")

        db.commit()

    print("\nDone. /health/models should now return 200.")


if __name__ == "__main__":
    register()