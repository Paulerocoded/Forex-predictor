# forex_app_v2.py
# Smart Forex Predictor v2
# Built for Streamlit Cloud Deployment (with Telegram/Email alerts)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests
import smtplib
from email.mime.text import MIMEText

st.set_page_config(page_title="Smart Forex Predictor v2", page_icon="💹", layout="centered")

st.title("💹 Smart Forex Predictor v2")
st.caption("AI-powered Forex trend prediction and alert system")

# -----------------------------
# 1. Load Forex Dataset
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload your Forex CSV data (with Date, Open, High, Low, Close columns):", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(data.head())

    # Basic validation
    required_cols = {"Open", "High", "Low", "Close"}
    if not required_cols.issubset(data.columns):
        st.error("CSV file must contain at least: Date, Open, High, Low, Close columns")
        st.stop()

    # -----------------------------
    # 2. Preprocessing
    # -----------------------------
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[["Open", "High", "Low", "Close"]])

    # Create supervised dataset (X, y)
    X, y = [], []
    window_size = 5
    for i in range(window_size, len(data_scaled)):
        X.append(data_scaled[i - window_size:i])
        y.append(data_scaled[i, 3])  # Predict Close
    X, y = np.array(X), np.array(y)

    # -----------------------------
    # 3. Model Selection
    # -----------------------------
    model_type = st.selectbox("Choose AI Model", ["LSTM Neural Network", "Random Forest"])

    if model_type == "LSTM Neural Network":
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X, y, epochs=20, batch_size=16, verbose=0)
        st.success("✅ LSTM model trained successfully!")
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X.reshape(X.shape[0], -1), y)
        st.success("✅ Random Forest model trained successfully!")

    # -----------------------------
    # 4. Make Prediction
    # -----------------------------
    last_data = data_scaled[-window_size:]
    last_data = np.expand_dims(last_data, axis=0)

    if model_type == "LSTM Neural Network":
        predicted_scaled = model.predict(last_data)
    else:
        predicted_scaled = model.predict(last_data.reshape(1, -1))

    predicted_price = scaler.inverse_transform(
        np.concatenate((np.zeros((1, 3)), np.array(predicted_scaled).reshape(-1, 1)), axis=1)
    )[0, 3]

    st.metric("Predicted Next Close Price", f"${predicted_price:.4f}")

    # -----------------------------
    # 5. Alerts via Secrets
    # -----------------------------
    st.subheader("📢 Send Alerts (Telegram / Email)")

    secrets = st.secrets

    telegram_token = secrets.get("TELEGRAM_TOKEN")
    telegram_chat = secrets.get("TELEGRAM_CHAT")
    smtp_host = secrets.get("SMTP_HOST")
    smtp_port = secrets.get("SMTP_PORT")
    smtp_user = secrets.get("SMTP_USER")
    smtp_pass = secrets.get("SMTP_PASS")
    email_to = secrets.get("EMAIL_TO")

    alert_msg = f"📈 Forex Alert: Predicted Close Price = ${predicted_price:.4f}"

    def send_telegram(msg):
        if not telegram_token or not telegram_chat:
            st.warning("⚠️ Telegram secrets not set.")
            return
        try:
            requests.get(f"https://api.telegram.org/bot{telegram_token}/sendMessage",
                         params={"chat_id": telegram_chat, "text": msg})
            st.success("✅ Telegram alert sent successfully!")
        except Exception as e:
            st.error(f"Telegram error: {e}")

    def send_email(msg):
        if not all([smtp_host, smtp_port, smtp_user, smtp_pass, email_to]):
            st.warning("⚠️ Email secrets not set.")
            return
        try:
            email_msg = MIMEText(msg)
            email_msg["Subject"] = "Forex Prediction Alert"
            email_msg["From"] = smtp_user
            email_msg["To"] = email_to

            with smtplib.SMTP_SSL(smtp_host, int(smtp_port)) as server:
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, email_to, email_msg.as_string())
            st.success("✅ Email alert sent successfully!")
        except Exception as e:
            st.error(f"Email error: {e}")

    if st.button("Send Telegram Alert"):
        send_telegram(alert_msg)

    if st.button("Send Email Alert"):
        send_email(alert_msg)

else:
    st.info("👆 Please upload your Forex CSV file to begin.")
