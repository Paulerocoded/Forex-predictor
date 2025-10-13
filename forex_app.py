# forex_app_v9.1.py
# Smart Forex Predictor v9.1 — Debug-Enhanced + Candlestick Fix + Buy/Sell Suggestion + Auto Refresh

import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from io import StringIO
import time
import datetime
import plotly.graph_objects as go

# -----------------------
# APP CONFIGURATION
# -----------------------
st.set_page_config(page_title="Smart Forex Predictor v9.1",
                   page_icon="💹", layout="wide")

st.markdown("<h1 style='text-align:center'>💹 Smart Forex Predictor App </h1>", unsafe_allow_html=True)
st.caption("AI-powered forex predictor with candlestick visualization and debug diagnostics.")

# -----------------------
# SIDEBAR CONTROLS
# -----------------------
refresh_interval = st.sidebar.slider("🔁 Auto-refresh interval (minutes)", 1, 30, 5)
st.sidebar.info("The app auto-refreshes for live updates. You can adjust this interval below.")
st.sidebar.markdown("---")

# -----------------------
# FOREX PAIRS
# -----------------------
valid_symbols = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "USD/CHF": "USDCHF=X",
    "NZD/USD": "NZDUSD=X"
}

# -----------------------
# FETCH LIVE PRICES (for ticker)
# -----------------------
@st.cache_data(ttl=60)
def fetch_live_prices():
    prices = {}
    for name, symbol in valid_symbols.items():
        try:
            df = yf.download(symbol, period="1d", interval="1m", progress=False)
            if not df.empty:
                prices[name] = float(df["Close"].iloc[-1])
        except Exception:
            continue
    return prices

# -----------------------
# DISPLAY LIVE TICKER
# -----------------------
st.subheader("📡 Live Forex Price Ticker")

prices = fetch_live_prices()
if prices:
    ticker_html = " | ".join([f"<b>{pair}</b>: {price:.5f}" for pair, price in prices.items()])
    st.markdown(f"<div style='background:#111;color:#bfffd1;padding:8px;border-radius:6px'><marquee behavior='scroll' direction='left' scrollamount='4'>{ticker_html}</marquee></div>", unsafe_allow_html=True)
else:
    st.warning("⚠️ Could not fetch live ticker data right now.")

# -----------------------
# USER INPUTS
# -----------------------
pair = st.selectbox("Select Forex Pair:", list(valid_symbols.keys()))
symbol = valid_symbols[pair]
period = st.selectbox("Select Period:", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
interval = st.selectbox("Select Interval:", ["1d", "1h", "30m"], index=0)

st.markdown("---")

# -----------------------
# FETCH HISTORICAL DATA
# -----------------------
st.info(f"Fetching {pair} ({symbol}) historical data...")

try:
    data = yf.download(symbol, period=period, interval=interval, progress=False)
except Exception as e:
    st.error(f"❌ Failed to fetch data: {e}")
    st.stop()

if data.empty or len(data) < 5:
    st.error(f"⚠️ Data returned is empty or too short (rows={len(data)}). Try a longer period or different interval.")
    st.stop()

# -----------------------
# DEBUG DATA SNAPSHOT
# -----------------------
st.write("✅ Data successfully fetched.")
st.write(f"Rows: {len(data)}, Columns: {list(data.columns)}")
st.dataframe(data.tail())

# -----------------------
# ENSURE VALID DATETIME INDEX
# -----------------------
if not isinstance(data.index, pd.DatetimeIndex):
    st.warning("⚠️ Index is not datetime. Attempting to convert...")
    data = data.reset_index()
    if "Date" in data.columns:
        data["Datetime"] = pd.to_datetime(data["Date"])
    elif "index" in data.columns:
        data["Datetime"] = pd.to_datetime(data["index"])
    else:
        data["Datetime"] = pd.date_range(end=datetime.datetime.now(), periods=len(data), freq="D")
else:
    data = data.reset_index().rename(columns={"index": "Datetime"})

# -----------------------
# MODEL TRAINING
# -----------------------
data["Target"] = data["Close"].shift(-1)
data = data.dropna()

X = data[["Open", "High", "Low", "Close"]]
y = data["Target"]

model = RandomForestRegressor(n_estimators=120, random_state=42)
model.fit(X, y)
data["Predicted_Close"] = model.predict(X)

latest_row = data.iloc[-1]
latest_features = np.array(latest_row[["Open", "High", "Low", "Close"]]).reshape(1, -1)
next_pred = float(model.predict(latest_features)[0])

# -----------------------
# METRICS
# -----------------------
mae = mean_absolute_error(y, data["Predicted_Close"])
r2 = r2_score(y, data["Predicted_Close"])
accuracy_est = max(0, (1 - mae / np.mean(y)) * 100)

# -----------------------
# BUY / SELL / HOLD LOGIC
# -----------------------
current_close = float(latest_row["Close"])
pred_change = (next_pred - current_close) / current_close
pred_change_pct = pred_change * 100

buy_threshold = 0.0006
sell_threshold = -0.0006

if pred_change >= buy_threshold:
    suggestion = "BUY"
    suggestion_color = "green"
elif pred_change <= sell_threshold:
    suggestion = "SELL"
    suggestion_color = "red"
else:
    suggestion = "HOLD"
    suggestion_color = "orange"

explanation = f"Predicted next close: {next_pred:.5f} | Current: {current_close:.5f} | Δ = {pred_change_pct:.3f}% → Suggestion: {suggestion}"

# -----------------------
# DISPLAY RESULTS
# -----------------------
st.subheader("📊 Prediction Summary")
st.metric(label=f"Next {interval} Close Prediction for {pair}", value=f"{next_pred:.5f}")

col1, col2, col3, col4 = st.columns([2,2,2,3])
col1.metric("📉 MAE", f"{mae:.6f}")
col2.metric("📈 R²", f"{r2:.4f}")
col3.metric("🎯 Est. Accuracy", f"{accuracy_est:.2f}%")
col4.markdown(f"<div style='padding:10px;text-align:center;border-radius:6px;background:#fff'><h3 style='color:{suggestion_color};margin:0'>{suggestion}</h3><small>{explanation}</small></div>", unsafe_allow_html=True)

# -----------------------
# FIXED CANDLESTICK CHART
# -----------------------
st.subheader("🕯️ Candlestick Chart (with Predicted Next Close)")

try:
    if len(data) < 2:
        st.warning("Not enough data to plot candles.")
    else:
        last_time = data["Datetime"].iloc[-1]
        if interval == "1d":
            next_time = last_time + pd.Timedelta(days=1)
        elif interval == "1h":
            next_time = last_time + pd.Timedelta(hours=1)
        else:
            next_time = last_time + pd.Timedelta(minutes=30)

        fig = go.Figure(data=[go.Candlestick(
            x=data["Datetime"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price"
        )])

        fig.add_trace(go.Scatter(
            x=[next_time],
            y=[next_pred],
            mode="markers+text",
            marker=dict(color=suggestion_color, size=12),
            text=[f"Pred: {next_pred:.5f}"],
            textposition="top center",
            name="Prediction"
        ))

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=520,
            template="plotly_dark",
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"⚠️ Error plotting candlestick: {e}")

# -----------------------
# DOWNLOAD REPORT
# -----------------------
latest_data = data.tail(1).copy()
latest_data["Next_Predicted_Close"] = next_pred
latest_data["Currency_Pair"] = pair
latest_data["Interval"] = interval
latest_data["MAE"] = mae
latest_data["R2_Score"] = r2
latest_data["Accuracy(%)"] = accuracy_est
latest_data["Suggestion"] = suggestion
latest_data["Predicted_Change(%)"] = pred_change_pct
latest_data["Timestamp"] = datetime.datetime.now()

csv_buffer = StringIO()
latest_data.to_csv(csv_buffer, index=False)
st.download_button(
    label="📥 Download Prediction Report (CSV)",
    data=csv_buffer.getvalue().encode("utf-8"),
    file_name=f"{pair.replace('/', '-')}_prediction_report.csv",
    mime="text/csv"
)

st.caption("⚡ Powered by Yahoo Finance & Random Forest — Use predictions for education, not real trades.")

# -----------------------
# AUTO REFRESH
# -----------------------
next_refresh = datetime.datetime.now() + datetime.timedelta(minutes=refresh_interval)
st.sidebar.info(f"⏳ Auto-refreshing at {next_refresh.strftime('%H:%M:%S')}...")
time.sleep(refresh_interval * 60)
st.experimental_rerun()
