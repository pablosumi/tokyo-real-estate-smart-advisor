# Tokyo Real Estate Smart Advisor
Machine learning pipeline and Streamlit dashboard with LLM interactivity for Tokyo residential price intelligence. It estimates market value with an XGBoost model, shows historical transaction prices, and lets you chat with an AI advisor via the OpenRouter API.

## Highlights
- Data ingestion from the MLIT Real Estate Information Library, plus cleaning and feature engineering tailored to residential properties.
- XGBoost training pipeline with target encoding, logged health checks, and packaged artifacts (`models/tokyo_mass_market_xgb.pkl`).
- Streamlit app (`dashboard.py`) that loads the trained model to estimate prices, charts transaction history, and provides an OpenRouter-powered chat assistant.
- Parquet-based data flow: raw -> cleaned -> model-ready. Logs, data, and model artifacts live in git-ignored folders.

## Quick start
1) Python 3.11+ recommended. Create a virtualenv and install dependencies:
```bash
pip install -r requirements.txt
```
2) Add a `.env` file with required keys:
```bash
MLIT_API_KEY=your_mlit_key        # needed for data ingestion from MLIT
OPENROUTER_API_KEY=your_key       # needed for dashboard chat feature
```
3) Build data and the model (outputs land in `data/` and `models/`):
```bash
python scripts/ingest.py             # raw MLIT pulls -> data/tokyo.parquet
python scripts/clean.py              # cleaning/filtering -> data/tokyo-clean.parquet
python scripts/preprocessing_xgb.py  # feature engineering -> data/tokyo-preprocessed.parquet
python scripts/train_xgb.py          # trains & packages artifacts -> models/tokyo_mass_market_xgb.pkl
```

4) Run the Streamlit dashboard:
```bash
streamlit run dashboard.py
```
   - The app reads `data/tokyo-clean.parquet` for price charts and `models/tokyo_mass_market_xgb.pkl` for predictions.

## Directory structure
```
tokyo-real-estate-smart-advisor/
├── .streamlit/                       
│   └── config.toml                   # streamlit config
├── .venv/                            # git ignored (local Python virtual env)
├── data/                             # git ignored
│   ├── tokyo-clean.parquet           # cleaned MLIT data
│   ├── tokyo-preprocessed.parquet    # preprocessed MLIT data for XGBoost (stateless)
│   └── tokyo.parquet                 # raw MLIT data
├── logs/                             # git ignored
│   ├── clean.log                     # clean execution history (timestamps, row counts)
│   ├── ingest.log                    # ingest execution history (timestamps, row counts)
│   ├── preprocessing_xgb.log         # preprocessing execution history (timestamps, features)
│   └── train_xgb.log                 # xgb re-training history (timestamps, evals)
├── models/                           # git ignored
│   ├── best_hyperparameters_xgb.json
│   ├── model_history.csv             # history of re-trained models' eval metrics
│   └── tokyo_mass_market_xgb.pkl     # xgboost trained on all mass market data
├── notebooks/
│   ├── clean.ipynb                   # cleaning raw data
│   ├── EDA.ipynb                     # exploratory data analysis
│   ├── ingest.ipynb                  # ingesting raw data from MLIT
│   ├── modeling_xgb.ipynb            # XGBoost experimentation
│   └── preprocessing_xgb.ipynb       # stateless preprocsesing for XGBoost
├── scripts/
│   ├── clean.py                      # applies cleaning -> tokyo-clean.parquet
│   ├── ingest.py                     # streamed data pull from MLIT -> tokyo.parquet
│   ├── preprocessing_xgb.py          # adds features for xgb -> tokyo-preprocessed.parquet
│   └── train_xgb.py                  # xgb re-training pipeline -> tokyo_mass_market_xgb.pkl
├── src/
│   ├── __pycache__/                  # git ignored
│   ├── __init__.py
│   ├── api.py                        # MLIT API wrapper (auth, data fetching)                  
│   ├── chat.py                       # OpenRouter LLM functionality for dashboard chatbox
│   ├── cleaning_utils.py             # cleaning logic
│   ├── config.py                     # project constants (URLs, defaults, pref codes, paths)
│   ├── features.py                   # feature engineering logic
│   └── inference.py                  # predict with tokyo_mass_market_xgb.pkl trained xgboost model
├── .env                              # git ignored (MLIT api key)
├── .gitattributes
├── .gitignore
├── dashboard.py                      # streamlit dashboard logic
├── README.md
└── requirements.txt                  # Python dependencies
```

## Repo layout
- `dashboard.py` — Streamlit UI for valuation, charts, and LLM chat.
- `scripts/` — ingestion, cleaning, preprocessing, and XGBoost training entry points.
- `src/` — API client, feature engineering, inference wrapper, and chat helper.
- `data/`, `models/`, `logs/` — git-ignored artifacts created by the pipelines.
- `notebooks/` — jupyter notebooks for ingestion, cleaning, EDA, and modeling.

## Notes
- Data source: Ministry of Land, Infrastructure, Transport and Tourism (MLIT) Real Estate Information Library (国土交通省不動産情報ライブラリ) (reinfolib) endpoint 4. (https://www.reinfolib.mlit.go.jp/help/apiManual/).
- The OpenRouter model used by default is set in `src/config.py`; override via the `model` parameter in `get_chat_completion` if desired.
