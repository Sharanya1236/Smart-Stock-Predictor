import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import altair as alt
from PIL import Image

# --- Page config ---
st.set_page_config(page_title="StockSense", page_icon="📈", layout="wide")

# --- Header with logo ---
st.markdown("""
    <style>
    .main-title {
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1.2em;
        color: #bbbbbb;
    }
    .description {
        font-size: 0.95em;
        color: #dddddd;
        padding-bottom: 15px;
    }
    .block-container {
        padding: 2rem 2rem 2rem 2rem;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 6])
with col1:
    try:
        logo = Image.open("logo.jpeg")
        st.image(logo, width=80)
    except:
        st.image("https://img.icons8.com/fluency/96/stock-share.png", width=80)
with col2:
    st.markdown("<div class='main-title'>StockSense</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Your AI-powered stock forecasting companion</div>", unsafe_allow_html=True)

st.markdown("""
<div class='description'>
Easily select a country and company to forecast stock prices. Powered by historical market data and machine learning.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Company dropdowns by country ---
company_map = {
    "India (NSE)": {
        "RELIANCE": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "INFY": "INFY.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "ICICI Bank": "ICICIBANK.NS"
    },
    "US (NYSE/NASDAQ)": {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Google": "GOOGL",
        "Amazon": "AMZN",
        "Tesla": "TSLA"
    },
    "UK (LSE)": {
        "HSBC": "HSBA.L",
        "BP": "BP.L",
        "GlaxoSmithKline": "GSK.L",
        "Barclays": "BARC.L",
        "Vodafone": "VOD.L"
    }
}

# --- Sidebar ---
st.sidebar.header("📊 Configuration")
selected_country = st.sidebar.selectbox("Select Country/Market", list(company_map.keys()))
company_list = list(company_map[selected_country].keys())
selected_company = st.sidebar.selectbox("Select Company", company_list)
ticker = company_map[selected_country][selected_company]
days = st.sidebar.slider("Days to Predict", 1, 10, 5)

# --- Predict Button ---
st.markdown("### 📈 Prediction Tool")
if st.button("🔮 Predict Next Days"):
    st.info(f"Fetching and predicting data for **{selected_company} ({ticker})**...")

    data = yf.download(ticker, period="3y", interval="1d")
    if data.empty:
        st.error("No data found for this company. Please try another.")
    else:
        # --- Show historical chart ---
        st.subheader("Historical Closing Prices")
        hist_df = data.reset_index()
        hist_chart = alt.Chart(hist_df).mark_line().encode(
            x='Date:T',
            y='Close:Q',
            tooltip=['Date:T', 'Close:Q']
        ).properties(
            width=800,
            height=350
        )
        st.altair_chart(hist_chart, use_container_width=True)

        # --- Train model ---
        df = data[['Close']].copy()
        df['Day'] = np.arange(len(df))
        X = df[['Day']]
        y = df['Close']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # --- Predict ---
        last_day = df['Day'].iloc[-1]
        future_days = np.arange(last_day + 1, last_day + 1 + days).reshape(-1, 1)
        preds = model.predict(future_days)

        future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, days + 1)]
        pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': preds})

        st.subheader("📅 Predicted Prices")
        st.dataframe(pred_df.style.format({"Predicted Close": "{:.2f}"}), use_container_width=True)

        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions", csv, f"{selected_company}_predictions.csv", "text/csv")

        # --- Show predicted chart ---
        pred_chart = alt.Chart(pred_df).mark_line(color='orange').encode(
            x='Date:T',
            y='Predicted Close:Q',
            tooltip=['Date:T', 'Predicted Close:Q']
        ).properties(
            width=800,
            height=350
        )
        st.altair_chart(pred_chart, use_container_width=True)

st.markdown("---")

# --- Footer ---
st.markdown("""
    <div style='text-align:center; font-size:13px; color: gray;'>
        Developed by <b>Sharanya Godishala</b> | 
        <a href='https://github.com/Sharanya1236' target='_blank'>GitHub</a> | 
        <a href='https://www.linkedin.com/in/sharanya-godishala-a16998313/' target='_blank'>LinkedIn</a>
    </div>
""", unsafe_allow_html=True)
