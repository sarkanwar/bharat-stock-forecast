import os, re, requests, numpy as np, pandas as pd, streamlit as st, yfinance as yf
from datetime import datetime, timedelta
from PyPDF2 import PdfReader
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="India Stock Predictor (NSE/BSE)", layout="wide")
st.title("🇮🇳 India Stock Predictor — NSE / BSE")
st.caption("Upload a PDF or type names/tickers. Uses FMP API (if provided) + Yahoo for Indian markets. Educational only.")

# =========================
# Secrets / Settings
# =========================
FMP_API_KEY = st.secrets.get("FMP_API_KEY", None)

# =========================
# Helpers
# =========================
def yahoo_search_symbol(name, lang="en-US", region="IN"):
    """Best-effort name->symbol via Yahoo search (free)."""
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        r = requests.get(url, params={"q": name, "lang": lang, "region": region}, timeout=10)
        r.raise_for_status()
        data = r.json()
        for q in data.get("quotes", []):
            if q.get("quoteType") in {"EQUITY","ETF"} and q.get("symbol"):
                return q["symbol"]
    except Exception:
        pass
    return None

def fmp_search_symbol(query, exchange_hint="NSE"):
    """Prefer FMP global search; fallback to Yahoo if no key or miss."""
    # Try FMP
    if FMP_API_KEY:
        try:
            url = "https://financialmodelingprep.com/api/v3/search"
            params = {"query": query, "limit": 3, "apikey": FMP_API_KEY}
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                arr = r.json()
                # prioritize by exchange hint if present in result (FMP often returns .NS / .BO already)
                for item in arr:
                    sym = item.get("symbol", "")
                    ex = (item.get("exchangeShortName") or item.get("exchange") or "").upper()
                    if exchange_hint.upper() in (ex or "") or sym.endswith(".NS") or sym.endswith(".BO"):
                        return sym
                if arr:
                    return arr[0].get("symbol")
        except Exception:
            pass
    # Fallback to Yahoo
    return yahoo_search_symbol(query, region="IN")

def ensure_india_suffix(symbol, exchange):
    """Append .NS or .BO if missing; leave if already present."""
    if not isinstance(symbol, str) or not symbol:
        return None
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    suffix = ".NS" if exchange == "NSE" else ".BO"
    return symbol + suffix

def extract_candidates_from_pdf(file_like):
    reader = PdfReader(file_like)
    text = " ".join([(p.extract_text() or "") for p in reader.pages])
    tickers = set(re.findall(r"\b[A-Z]{1,10}\b", text))
    words = set(re.findall(r"\b[A-Za-z][A-Za-z\.\-& ]{2,}\b", text))
    candidates = list(tickers | {w.strip() for w in words if len(w.strip()) <= 40})
    blacklist = {"USD","NSE","BSE","NYSE","NASDAQ","LTD","INC","PVT","AND","THE","OF","IN","ON","BY","LIMITED","PRIVATE"}
    return [c for c in candidates if c.upper() not in blacklist]

def parse_manual(text):
    if not text: return []
    raw = [x.strip() for x in text.replace("\n", ",").split(",")]
    return [x for x in raw if x]

def get_history(symbol, years=8):
    end = datetime.utcnow()
    start = end - timedelta(days=365*years + 30)
    df = yf.download(symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    df = df.reset_index()[["Date","Open","High","Low","Close","Volume"]]
    df.columns = ["date","open","high","low","close","volume"]
    df.dropna(inplace=True)
    return df

def add_features(df):
    df = df.copy()
    df["ret_1d"] = df["close"].pct_change()
    df["log_ret"] = np.log1p(df["ret_1d"])
    # moving avgs
    for n in [5,10,20,50,100,200]:
        df[f"sma_{n}"] = df["close"].rolling(n).mean()
        df[f"ema_{n}"] = df["close"].ewm(span=n, adjust=False).mean()
    # RSI 14
    delta = df["close"].diff(); up = delta.clip(lower=0); down = -delta.clip(upper=0)
    roll_up = up.ewm(com=13, adjust=False).mean(); roll_down = down.ewm(com=13, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9); df["rsi_14"] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    # Bollinger
    ma20 = df["close"].rolling(20).mean(); sd20 = df["close"].rolling(20).std()
    df["bb_mid"] = ma20; df["bb_upper"] = ma20 + 2*sd20; df["bb_lower"] = ma20 - 2*sd20
    # Vol & lags
    df["vol_20"] = df["ret_1d"].rolling(20).std()
    for l in [1,2,3,5,10]:
        df[f"lag_{l}"] = df["ret_1d"].shift(l)
    df.dropna(inplace=True)
    return df

def train_predict(df_feat, horizon_days):
    # target future close
    y_future = df_feat["close"].shift(-horizon_days)
    data = df_feat.iloc[:-horizon_days].copy()
    data["target"] = y_future.iloc[:-horizon_days]
    feats = [c for c in data.columns if c not in ["date","open","high","low","volume","target"]]
    X, y = data[feats], data["target"]
    if len(X) < 200:
        raise ValueError("Not enough samples to train")
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    pred = float(model.predict(df_feat[feats].iloc[[-1]])[0])
    return pred

def simple_backtest(df_feat, horizon_days):
    df = df_feat.copy().reset_index(drop=True)
    if "sma_20" not in df.columns:
        df["sma_20"] = df["close"].rolling(20).mean()
    df["signal"] = ((df["close"] > df["sma_20"]) & (df["rsi_14"].diff() > 0)).astype(int)
    df["fwd_ret"] = df["close"].shift(-horizon_days)/df["close"] - 1.0
    df["strategy_ret"] = df["signal"] * df["fwd_ret"]
    df["equity"] = (1.0 + df["strategy_ret"].fillna(0)).cumprod()
    return df

# =========================
# UI
# =========================
with st.sidebar:
    st.header("Inputs (India)")
    exchange = st.selectbox("Exchange", options=["NSE","BSE"], index=0, help="NSE adds .NS, BSE adds .BO to tickers")
    pdf_file = st.file_uploader("Upload PDF with company names/tickers", type=["pdf"])
    manual = st.text_area("Or enter names/tickers (comma/newline)")
    run_btn = st.button("Run")

cands = []
if pdf_file:
    try:
        cands += extract_candidates_from_pdf(pdf_file)
        st.success(f"PDF parsed. Found {len(cands)} candidates.")
    except Exception as e:
        st.error(f"PDF parse error: {e}")
cands += parse_manual(manual)
cands = list(dict.fromkeys(cands))

if cands:
    st.write("Candidates:", ", ".join(cands[:50]) + (" ..." if len(cands)>50 else ""))

if run_btn:
    if not cands:
        st.warning("Please upload a PDF or enter names/tickers.")
        st.stop()

    with st.spinner("Resolving names to Indian tickers..."):
        mapping = {}
        for n in cands:
            if n.isupper() and len(n) <= 10 and not n.endswith((".NS",".BO")):
                mapping[n] = ensure_india_suffix(n, exchange)
            else:
                sym = fmp_search_symbol(n, exchange_hint=exchange)
                mapping[n] = ensure_india_suffix(sym, exchange) if sym else None

    st.subheader("Resolved Tickers")
    st.dataframe(pd.DataFrame(mapping.items(), columns=["Input","Ticker"]))

    tickers = [t for t in mapping.values() if t]
    if not tickers:
        st.error("No valid Indian tickers resolved. Try entering the official NSE/BSE code (e.g., RELIANCE, TCS, INFY).")
        st.stop()

    horizons = {"1D":1,"1W":7,"1M":30,"6M":180,"1Y":365,"3Y":365*3,"5Y":365*5}
    rows = []
    for t in tickers:
        with st.spinner(f"Fetching & modeling {t}..."):
            hist = get_history(t, years=8)
            if hist is None or len(hist) < 300:
                st.warning(f"{t}: insufficient history or Yahoo symbol mismatch. Skipping.")
                continue
            feat = add_features(hist)
            row = {"Ticker": t, "CurrentPrice": float(hist["close"].iloc[-1])}
            for label, days in horizons.items():
                try:
                    row[label] = train_predict(feat, days)
                except Exception as e:
                    row[label] = None
            rows.append(row)

    if rows:
        dfp = pd.DataFrame(rows)
        st.subheader("Predicted Prices (INR)")
        st.dataframe(dfp.set_index("Ticker").round(2))
        st.download_button("Download CSV", data=dfp.to_csv(index=False).encode("utf-8"),
                           file_name="predictions_india.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Backtest (rule-based)")
        sel = st.selectbox("Ticker", options=[r["Ticker"] for r in rows])
        horizon_choice = st.selectbox("Horizon", options=list(horizons.keys()), index=2)
        lookback = st.slider("Lookback years", 2, 8, 5)
        if st.button("Run Backtest"):
            hist = get_history(sel, years=lookback)
            if hist is None:
                st.error("No history for backtest.")
            else:
                feat = add_features(hist)
                bt = simple_backtest(feat, horizons[horizon_choice])
                st.line_chart(bt.set_index("date")["equity"])

st.markdown("---")
note = "Tip: For NSE, use raw codes like RELIANCE, TCS, INFY — the app will auto-append .NS. For BSE, it appends .BO."
st.caption(note)
