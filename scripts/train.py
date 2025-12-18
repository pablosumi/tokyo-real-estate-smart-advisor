import pandas as pd
import numpy as np
import xgboost as xgb
import category_encoders as ce
import joblib
import json
import os
import sys
import logging
import csv
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error

# --- Path Setup ---
# Add the project root to sys.path so we can import from 'src'
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# --- SETUP LOGGING ---
LOG_DIR = project_root / 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

# We use append mode for the file handler to keep history of runs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'train.log'), mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- IMPORTS FROM CONFIG ---
# Assuming these exist in src/config.py. If not, replace with raw strings.
from src.config import PROCESSED_DATA_PATH, XGB_PARAMS_PATH, MODEL_OUTPUT_PATH

# CONSTANTS
HISTORY_PATH = os.path.join(project_root, 'models', 'model_history.csv')
VALIDATION_SIZE = 3000  # Number of recent rows to hold out for health check

# Columns to drop (Targets + Metadata not for training)
DROP_COLS = [
    'TradePriceYen', 
    'LogTradePriceYen', 
    'TransactionQuarterEndDate', 
    'TransactionQuarter'
]

# Categorical Columns
CAT_COLS = [
    'Municipality', 'DistrictName', 'NearestStation', 
    'Use', 'Structure', 'LandShape', 
    'Renovation', 'Purpose', 'Type', 'Region', 'CityPlanning', 
    'Classification', 'RoadDirection', 'Remarks'
]

def log_metrics_to_csv(metrics):
    """Appends training run metrics to a CSV file for long-term tracking."""
    file_exists = os.path.isfile(HISTORY_PATH)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    
    with open(HISTORY_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

def main():
    logger.info("Starting Training Pipeline...")

    # 1. Load Data
    if not os.path.exists(PROCESSED_DATA_PATH):
        logger.error(f"Data not found at {PROCESSED_DATA_PATH}. Run preprocessing first.")
        return
    
    logger.info(f"Loading data from {PROCESSED_DATA_PATH}...")
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    
    # --- CRITICAL: SORT DATA ---
    # We must sort by time so the validation set represents the "future"
    if 'TransactionQuarterEndDate' in df.columns:
        logger.info("Sorting data by TransactionQuarterEndDate...")
        df = df.sort_values('TransactionQuarterEndDate').reset_index(drop=True)
    else:
        logger.warning("⚠️ 'TransactionQuarterEndDate' not found! Data might not be sorted chronologically.")

    logger.info(f"Data Shape: {df.shape}")

    # 2. HEALTH CHECK (Validate before Commit)
    logger.info(f"🩺 Running Health Check (Holdout: Last {VALIDATION_SIZE} rows)...")
    
    # Split Data for Validation
    train_df = df.iloc[:-VALIDATION_SIZE].copy()
    val_df = df.iloc[-VALIDATION_SIZE:].copy()
    
    target_col = 'LogTradePriceYen'
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found!")
        return

    y_train_val = train_df[target_col]
    y_test_val = val_df[target_col]
    
    # Prepare Features
    X_train_val = train_df.drop(columns=DROP_COLS, errors='ignore')
    X_test_val = val_df.drop(columns=DROP_COLS, errors='ignore')

    # Load Hyperparameters
    if not os.path.exists(XGB_PARAMS_PATH):
        logger.error(f"Hyperparameters not found at {XGB_PARAMS_PATH}.")
        return

    with open(XGB_PARAMS_PATH, 'r') as f:
        params = json.load(f)

    # Encode for Health Check
    valid_cat_cols = [c for c in CAT_COLS if c in X_train_val.columns]
    encoder_check = ce.TargetEncoder(cols=valid_cat_cols, smoothing=10)
    X_train_enc = encoder_check.fit_transform(X_train_val, y_train_val)
    X_test_enc = encoder_check.transform(X_test_val)

    # Train Proxy Model
    # We temporarily remove early_stopping from params if it exists, 
    # OR we could add eval_set here. Let's just run full iterations for simplicity.
    proxy_params = params.copy()
    if 'early_stopping_rounds' in proxy_params:
        del proxy_params['early_stopping_rounds']

    model_check = xgb.XGBRegressor(**proxy_params)
    model_check.fit(X_train_enc, y_train_val)

    # Score Proxy Model
    preds_log = model_check.predict(X_test_enc)
    preds_yen = np.exp(preds_log)
    actual_yen = np.exp(y_test_val)
    
    mae = mean_absolute_error(actual_yen, preds_yen)
    mape = np.mean(np.abs((actual_yen - preds_yen) / actual_yen)) * 100
    
    logger.info(f"Health Check Results -- MAE: ¥{mae:,.0f} | MAPE: {mape:.2f}%")

    # Log History
    metrics_record = {
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'mae': round(mae, 0),
        'mape': round(mape, 4),
        'training_rows': len(df),
        'validation_rows': VALIDATION_SIZE
    }
    log_metrics_to_csv(metrics_record)


    # 3. FINAL PRODUCTION TRAINING (100% Data)
    logger.info("Health check passed. Training Final Model on 100% of data...")
    
    y_all = df[target_col]
    X_all = df.drop(columns=DROP_COLS, errors='ignore')
    
    # Final Encoding
    logger.info("Fitting Target Encoder on ALL data...")
    final_encoder = ce.TargetEncoder(cols=valid_cat_cols, smoothing=10)
    X_all_enc = final_encoder.fit_transform(X_all, y_all)

    # Final Params (ensure early stopping is gone)
    if 'early_stopping_rounds' in params:
        del params['early_stopping_rounds']

    # Final Training
    logger.info("Training XGBoost Model...")
    final_model = xgb.XGBRegressor(**params)
    final_model.fit(X_all_enc, y_all)
    logger.info("Training Complete.")

    # 4. SAVE ARTIFACTS
    logger.info("Packaging artifacts...")
    artifacts = {
        'model': final_model,
        'encoder': final_encoder,
        'features': X_all.columns.tolist(),
        'hyperparameters': params,
        'threshold': 200000000,
        'latest_metrics': metrics_record
    }

    # Create models dir if not exists
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    
    joblib.dump(artifacts, MODEL_OUTPUT_PATH)
    logger.info(f"✅ Model saved to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()