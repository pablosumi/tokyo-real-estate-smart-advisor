# Tokyo Real Estate Smart Advisor

Coming soon.

## Directory structure
```
tokyo-real-estate-smart-advisor/
├── README.md
├── env.json               
├── .gitignore
├── data/
│   ├── tokyo.parquet      
│   └── tokyo-clean.parquet
└── notebooks/
    ├── ingest.ipynb
    ├── clean.ipynb
    └── EDA.ipynb
```

## Environment configuration
- Create an `env.json` in the project root (already git-ignored) with your MLIT API key.
- Example structure (replace the placeholder with your key):

```json
{
  "MLIT_API_KEY": "your-mlit-api-key"
}
```

## Data
- Source: Ministry of Land, Infrastructure, Transport and Tourism (MLIT) Real Estate Information Library (国土交通省不動産情報ライブラリ) (`reinfolib`).
- API docs: https://www.reinfolib.mlit.go.jp/help/apiManual/ (endpoint 4).
- Data artifacts live in `data/` as parquet exports.