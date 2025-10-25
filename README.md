# India Stock Predictor — NSE/BSE (Single File)

## Features
- PDF upload + manual entry
- Indian ticker resolver: auto-appends .NS (NSE) or .BO (BSE)
- Uses FMP API (if provided via Streamlit Secrets) for robust search
- Historical data via yfinance
- Multi-horizon price predictions
- Simple backtesting

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Secrets (optional)
Add in Streamlit Cloud -> Manage App -> Secrets:
```
FMP_API_KEY = "your_free_fmp_key_here"
```
