# src/config.py

# ingest.py
BASE_URL = "https://www.reinfolib.mlit.go.jp/ex-api/external/XIT001"
PREF_CODE = "13"  # Tokyo
START_YEAR = 2010
TIMEOUT = 30

# train.py
PROCESSED_DATA_PATH = 'data/tokyo-preprocessed.parquet'
XGB_PARAMS_PATH = 'models/best_hyperparameters_xgb.json'
MODEL_OUTPUT_PATH = 'models/tokyo_mass_market_xgb.pkl'


# chat.py
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nex-agi/deepseek-v3.1-nex-n1:free"
SYSTEM_PROMPT = (
    "You are a Tokyo residential real estate market advisor."
    "Use the provided property details and recent market behavior to answer clearly, "
    "note uncertainty when data is thin, and do not fabricate numbers."
)