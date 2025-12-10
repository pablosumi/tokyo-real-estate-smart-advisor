# Tokyo Real Estate Smart Advisor

Coming soon.

## Directory structure
```
tokyo-real-estate-smart-advisor/
├── .venv/                         # git ignored (local py virtual env)
├── data/                          # git ignored
│   ├── tokyo.parquet              # raw MLIT data
│   └── tokyo-clean.parquet        # cleaned MLIT data
├── notebooks/
│   ├── ingest.ipynb
│   ├── clean.ipynb
│   └── EDA.ipynb
├── scripts/
│   └── ingest.py                  # streamed data pull from MLIT -> tokyo.parquet
├── .env                           # git ignored (MLIT api key)
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt               # Python dependencies
```

## Data
- Source: Ministry of Land, Infrastructure, Transport and Tourism (MLIT) Real Estate Information Library (国土交通省不動産情報ライブラリ) (`reinfolib`) endpoint 4.
- API docs: https://www.reinfolib.mlit.go.jp/help/apiManual/
- Data artifacts live in `data/` as parquet exports.
