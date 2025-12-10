# Tokyo Real Estate Smart Advisor

Coming soon.

## Directory structure
```
tokyo-real-estate-smart-advisor/
├── README.md
├── .env                           # git ignored       
├── .gitignore
├── data/                          # git ignored
│   ├── tokyo.parquet              # raw MLIT data
│   └── tokyo-clean.parquet        # cleaned MLIT data
└── notebooks/
    ├── ingest.ipynb
    ├── clean.ipynb
    └── EDA.ipynb
```

## Data
- Source: Ministry of Land, Infrastructure, Transport and Tourism (MLIT) Real Estate Information Library (国土交通省不動産情報ライブラリ) (`reinfolib`).
- API docs: https://www.reinfolib.mlit.go.jp/help/apiManual/ (endpoint 4).
- Data artifacts live in `data/` as parquet exports.