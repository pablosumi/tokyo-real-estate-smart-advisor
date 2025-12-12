# Tokyo Real Estate Smart Advisor

Work in progress. Tokyo Real Estate Intelligence: ML Prediction & AI Assistant.

## Directory structure
```
tokyo-real-estate-smart-advisor/
├── .venv/                          # git ignored (local Python virtual env)
├── data/                           # git ignored
│   ├── tokyo-clean.parquet         # cleaned MLIT data
│   ├── tokyo-preprocessed.parquet  # preprocessed MLIT data for XGBoost (stateless)
│   └── tokyo.parquet               # raw MLIT data
├── notebooks/
│   ├── clean.ipynb                 # cleaning raw data
│   ├── EDA.ipynb                   # exploratory data analysis
│   ├── ingest.ipynb                # ingesting raw data from MLIT
│   └── preprocessing_xgb.ipynb     # stateless preprocsesing for XGBoost
├── scripts/
│   └── ingest.py                   # streamed data pull from MLIT -> tokyo.parquet
├── .env                            # git ignored (MLIT api key)
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt                # Python dependencies
```

## Data
- Source: Ministry of Land, Infrastructure, Transport and Tourism (MLIT) Real Estate Information Library (国土交通省不動産情報ライブラリ) (`reinfolib`) endpoint 4.
- API docs: https://www.reinfolib.mlit.go.jp/help/apiManual/
- Data artifacts live in `data/` as parquet exports.
