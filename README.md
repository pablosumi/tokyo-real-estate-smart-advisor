# Tokyo Real Estate Smart Advisor

Work in progress. Tokyo Real Estate Intelligence: ML Prediction & AI Assistant.

## Directory structure
```
tokyo-real-estate-smart-advisor/
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
│   ├── cleaning_utils.py             # cleaning logic
│   ├── config.py                     # project constants (URLs, defaults, pref codes, paths)
│   ├── features.py                   # feature engineering logic
│   └── inference.py                  # predict with tokyo_mass_market_xgb.pkl trained xgboost model
├── .env                              # git ignored (MLIT api key)
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt                  # Python dependencies
```

## Data
- Source: Ministry of Land, Infrastructure, Transport and Tourism (MLIT) Real Estate Information Library (国土交通省不動産情報ライブラリ) (`reinfolib`) endpoint 4.
- API docs: https://www.reinfolib.mlit.go.jp/help/apiManual/
- Data artifacts live in `data/` as parquet exports.
