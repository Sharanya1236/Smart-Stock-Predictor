import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from PIL import Image
import altair as alt
import os

# --- Page config ---
st.set_page_config(page_title="StockSense", page_icon="📈", layout="centered")

# --- Load logo safely ---
logo_path = "assets/logo.jpeg"
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
else:
    logo = None  # fallback if image is missing

# --- Header ---
col1, col2 = st.columns([1, 6])
with col1:
    if logo:
        st.image(logo, width=80)
with col2:
    st.markdown(
        """
        <h1 style='margin-bottom: 0; padding-top: 10px;'>StockSense</h1>
        <p style='font-size: 18px; color: grey; margin-top: 2px;'>A smart stock predictor</p>
        """,
        unsafe_allow_html=True
    )


st.markdown("---")

# --- Sidebar inputs ---
st.sidebar.header("Settings")
market = st.sidebar.selectbox("Select Market", ["US (NYSE/NASDAQ)", "India (NSE)", "UK (LSE)"])
default_tickers = {
    "US (NYSE/NASDAQ)": "AAPL",
    "India (NSE)": "RELIANCE.NS",
    "UK (LSE)": "HSBA.L"
}
ticker = st.sidebar.text_input("Enter Ticker Symbol", value=default_tickers[market])
days = st.sidebar.slider("Days to Predict", 1, 5, 5)

# --- Styled button CSS ---
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Main app ---

if st.button("Predict Next Days"):

    if ticker.strip() == "":
        st.error("Please enter a ticker symbol.")
    else:
        with st.spinner("Fetching data and predicting..."):
            data = yf.download(ticker, period="3y", interval="1d")

            if data.empty:
                st.error("No data found for ticker. Check ticker symbol and market.")
            else:
                st.subheader(f"Historical Closing Prices for {ticker}")
                
                # Altair historical line chart
                hist_df = data.reset_index()
                hist_chart = alt.Chart(hist_df).mark_line().encode(
                    x='Date:T',
                    y='Close:Q',
                    tooltip=['Date:T', 'Close:Q']
                ).properties(
                    width=700,
                    height=350,
                    title='Historical Closing Prices'
                )
                st.altair_chart(hist_chart, use_container_width=True)

                # Prepare data for model
                df = data[['Close']].copy()
                df['Day'] = np.arange(len(df))
                X = df[['Day']]
                y = df['Close']

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)

                last_day = df['Day'].iloc[-1]
                future_days = np.arange(last_day + 1, last_day + 1 + days).reshape(-1, 1)
                preds = model.predict(future_days)

                future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, days + 1)]
                pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': preds}).set_index('Date')

                st.subheader(f"Predicted Closing Prices for Next {days} Day(s)")
                st.table(pred_df.style.format("{:.2f}"))

                # Altair predicted prices chart
                pred_chart = alt.Chart(pred_df.reset_index()).mark_line(color='orange').encode(
                    x='Date:T',
                    y='Predicted Close:Q',
                    tooltip=['Date:T', 'Predicted Close:Q']
                ).properties(
                    width=700,
                    height=350,
                    title='Predicted Closing Prices'
                )
                st.altair_chart(pred_chart, use_container_width=True)

                # Combine historical + predicted plots (matplotlib) for a quick glance
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10,5))
                plt.plot(data.index, data['Close'], label='Historical')
                plt.plot(pred_df.index, pred_df['Predicted Close'], label='Predicted', marker='o')
                plt.title(f"{ticker} Closing Price Prediction")
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.legend()
                plt.grid(True)
                st.pyplot(plt)

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <p style='text-align:center; font-size:12px; color:gray;'>
    Developed by Sharanya Godishala | 
    <a href="https://github.com/Sharanya1236" target="_blank">GitHub</a> | 
    <a href="https://www.linkedin.com/in/sharanya-godishala-a16998313/" target="_blank">LinkedIn</a>
    </p>
    """, unsafe_allow_html=True
)
