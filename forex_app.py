# forex_app_v9.py
# Smart Forex Predictor v9 — Candlestick + Buy/Sell suggestion + Live ticker + Auto-refresh

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
# APP CONFIG
# -----------------------
st.set_page_config(page_title="Smart Forex Predictor v9",
                   page_icon="💹", layout="wide")

st.markdown("<h1 style='text-align:center'>💹 Smart Forex Predictor App</h1>", unsafe_allow_html=True)
st.caption("AI-powered forecasting with candlestick chart and actionable Buy / Sell / Hold suggestion")

# -----------------------
# SIDEBAR CONTROLS
# -----------------------
refresh_interval = st.sidebar.slider("🔁 Auto-refresh interval (minutes)", 1, 30, 5)
ticker_speed = st.sidebar.slider("⏱️ Ticker update interval (seconds)", 5, 60, 15)
st.sidebar.info("Keep this tab open for live updates. The app will auto-reload every selected interval.")
st.sidebar.markdown("---")

# -----------------------
# SYMBOLS & TICKER
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

st.subheader("📡 Live Forex Ticker")
prices = fetch_live_prices()
if prices:
    ticker_html = " &nbsp; | &nbsp; ".join([f"<b>{pair}</b>: {price:.5f}" for pair, price in prices.items()])
    st.markdown(f"<div style='background:#111;color:#dfffd8;padding:8px;border-radius:6px'><marquee behavior=\"scroll\" direction=\"left\" scrollamount=\"4\">{ticker_html}</marquee></div>", unsafe_allow_html=True)
else:
    st.warning("Live ticker currently unavailable — try again in a moment.")

# -----------------------
# USER INPUTS
# -----------------------
pair = st.selectbox("Select Forex Pair to Predict:", list(valid_symbols.keys()))
symbol = valid_symbols[pair]

period = st.selectbox("Select data period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
interval = st.selectbox("Select interval:", ["1d", "1h", "30m"], index=0)

st.markdown("---")

# -----------------------
# FETCH HISTORICAL DATA
# -----------------------
st.info(f"Fetching historical data for {pair} ({symbol})...")
try:
    data = yf.download(symbol, period=period, interval=interval, progress=False)
    if data.empty:
        st.error("❌ No data found for this symbol/period. Try a different one.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error fetching data: {e}")
    st.stop()

st.success("✅ Data loaded")
st.dataframe(data.tail())

# -----------------------
# PREPARE DATA & MODEL
# -----------------------
data = data.copy()
data["Target"] = data["Close"].shift(-1)
data = data.dropna()

X = data[["Open", "High", "Low", "Close"]]
y = data["Target"]

model = RandomForestRegressor(n_estimators=120, random_state=42)
model.fit(X, y)

# predicted on training set for trend analysis
data["Predicted_Close"] = model.predict(X)
latest_row = data.iloc[-1]
latest_features = np.array(latest_row[["Open", "High", "Low", "Close"]]).reshape(1, -1)
next_pred = float(model.predict(latest_features)[0])

# performance metrics
mae = mean_absolute_error(y, data["Predicted_Close"])
r2 = r2_score(y, data["Predicted_Close"])
accuracy_est = max(0, (1 - mae / np.mean(y)) * 100)

# -----------------------
# BUY / SELL / HOLD LOGIC
# -----------------------
current_close = float(latest_row["Close"])
pred_change = (next_pred - current_close) / current_close  # relative change
pred_change_pct = pred_change * 100

# thresholds — easily tunable
buy_threshold = 0.0006    # 0.06%  (approx 6 pips on EURUSD)
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

# also create a short textual explanation
explanation = (f"Predicted next close: {next_pred:.5f}. Current close: {current_close:.5f}. "
               f"Predicted change: {pred_change_pct:.3f}% → Suggestion: {suggestion}")

# -----------------------
# SHOW RESULTS
# -----------------------
st.subheader("📊 Prediction Result")
st.metric(label=f"Predicted Next {interval} Close for {pair}", value=f"{next_pred:.5f}")

col1, col2, col3, col4 = st.columns([2,2,2,3])
col1.metric("📉 MAE", f"{mae:.6f}")
col2.metric("📈 R²", f"{r2:.4f}")
col3.metric("🎯 Est. Acc.", f"{accuracy_est:.2f}%")
col4.markdown(f"<div style='padding:10px;border-radius:6px;background:#fafafa;text-align:center'><h3 style='color:{suggestion_color};margin:0'>{suggestion}</h3><small>{explanation}</small></div>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------
# CANDLESTICK CHART (Plotly)
# -----------------------
st.subheader("🕯️ Candlestick Chart (with Predicted Next Close)")

fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Price")])

# add predicted next close as a marker at the next time tick
# compute x position for next tick
try:
    last_index = data.index[-1]
    # determine next timestamp depending on interval
    if interval == "1d":
        next_index = last_index + pd.Timedelta(days=1)
    elif interval == "1h":
        next_index = last_index + pd.Timedelta(hours=1)
    elif interval == "30m":
        next_index = last_index + pd.Timedelta(minutes=30)
    else:
        next_index = last_index + pd.Timedelta(days=1)
except Exception:
    next_index = last_index

fig.add_trace(go.Scatter(
    x=[next_index],
    y=[next_pred],
    mode="markers+text",
    marker=dict(color="green" if suggestion=="BUY" else ("red" if suggestion=="SELL" else "orange"),
                size=12),
    text=[f"Pred: {next_pred:.5f}"],
    textposition="top center",
    name="Prediction"))

fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=520,
    margin=dict(l=10, r=10, t=30, b=10)
)

st.plotly_chart(fig, use_container_width=True)

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
latest_data["Timestamp"] = datetime.datetime.now()
latest_data["Suggestion"] = suggestion
latest_data["Predicted_Change_pct"] = pred_change_pct

csv_buffer = StringIO()
latest_data.to_csv(csv_buffer, index=False)
csv_bytes = csv_buffer.getvalue().encode("utf-8")

st.download_button(
    label="📥 Download Prediction Report (CSV)",
    data=csv_bytes,
    file_name=f"{pair.replace('/', '-')}_prediction_report.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption("⚡ Powered by Yahoo Finance & Random Forest AI — Use suggestions as informational only. Backtest heavily before any trading.")

# -----------------------
# AUTO REFRESH
# -----------------------
# show next refresh time in sidebar
next_refresh_time = datetime.datetime.now() + datetime.timedelta(minutes=refresh_interval)
st.sidebar.write(f"Next refresh: {next_refresh_time.strftime('%Y-%m-%d %H:%M:%S')} (local)")

# pause and rerun to auto-refresh
time.sleep(refresh_interval * 60)
st.experimental_rerun()
