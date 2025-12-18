# src/config.py

BASE_URL = "https://www.reinfolib.mlit.go.jp/ex-api/external/XIT001"
PREF_CODE = "13"  # Tokyo
START_YEAR = 2010
TIMEOUT = 30
PROCESSED_DATA_PATH = 'data/tokyo-preprocessed.parquet'
XGB_PARAMS_PATH = 'models/best_hyperparameters_xgb.json'
MODEL_OUTPUT_PATH = 'models/tokyo_mass_market_xgb.pkl'