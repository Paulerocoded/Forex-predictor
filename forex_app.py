import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go

st.set_page_config(page_title="Forex Predictor", layout="wide")

st.title("💹 AI Forex Predictor App v9.1")

# Sidebar configuration
st.sidebar.header("⚙️ Settings")
pair = st.sidebar.text_input("Currency Pair (e.g. EURUSD=X, GBPUSD=X)", value="EURUSD=X")
period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y", "2y"])
interval = st.sidebar.selectbox("Interval", ["30m", "1h", "1d"])
st.sidebar.write("Example pairs: EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X")

# Load Data
@st.cache_data
def load_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval)
    if df.empty:
        return None
    df = df.dropna().copy()
    df.reset_index(inplace=True)
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df.rename(columns={"Adj Close": "Close"}, inplace=True)
    if "Datetime" not in df.columns:
        df.rename(columns={df.columns[0]: "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"])
    return df

data = load_data(pair, period, interval)

if data is None or data.empty:
    st.error("⚠️ No data found for this symbol. Try another one (e.g., EURUSD=X).")
    st.stop()

# Show data sample
st.subheader("📊 Recent Market Data")
st.dataframe(data.tail(10))

# Features
features = ["Open", "High", "Low", "Close"]
X = data[features]
y = data["Close"].shift(-1).dropna()
X = X.iloc[:-1, :]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
st.write(f"📉 Model MAE: **{mae:.5f}**")

# Next Prediction
last_row = data[features].iloc[-1].values.reshape(1, -1)
next_pred = model.predict(last_row)[0]

st.subheader("🔮 Next Predicted Close Price")
st.metric(label=f"Next Predicted Close for {pair}", value=f"{next_pred:.5f}")

# Trend Suggestion
last_close = data["Close"].iloc[-1]
if next_pred > last_close:
    suggestion = "Buy 📈"
    suggestion_color = "green"
elif next_pred < last_close:
    suggestion = "Sell 📉"
    suggestion_color = "red"
else:
    suggestion = "Hold ⚖️"
    suggestion_color = "gray"

st.markdown(f"### **Suggestion: <span style='color:{suggestion_color}'>{suggestion}</span>**", unsafe_allow_html=True)

# --- Candlestick Chart (fixed) ---
st.subheader("🕯️ Candlestick Chart")
try:
    plot_df = data.copy()
    plot_df["Datetime"] = pd.to_datetime(plot_df["Datetime"], errors="coerce")
    plot_df.dropna(subset=["Datetime"], inplace=True)

    fig = go.Figure(data=[go.Candlestick(
        x=plot_df["Datetime"],
        open=plot_df["Open"],
        high=plot_df["High"],
        low=plot_df["Low"],
        close=plot_df["Close"],
        name="Price"
    )])

    last_time = plot_df["Datetime"].iloc[-1]
    if interval == "1d":
        next_time = last_time + pd.Timedelta(days=1)
    elif interval == "1h":
        next_time = last_time + pd.Timedelta(hours=1)
    elif interval == "30m":
        next_time = last_time + pd.Timedelta(minutes=30)
    else:
        next_time = last_time + pd.Timedelta(days=1)

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
