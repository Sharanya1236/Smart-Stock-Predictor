import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import altair as alt
from PIL import Image

# --- Page config ---
st.set_page_config(page_title="StockSense", page_icon="💹", layout="wide")

# --- Hardcoded top companies by country (could be moved to a separate file) ---
top_companies_by_country = {
    "India (NSE)": {
        "Reliance Industries": "RELIANCE.NS",
        "Tata Consultancy Services": "TCS.NS",
        "Infosys": "INFY.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "ICICI Bank": "ICICIBANK.NS"
    },
    "US (NYSE/NASDAQ)": {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Amazon": "AMZN",
        "Google (Alphabet)": "GOOGL",
        "Tesla": "TSLA"
    },
    "UK (LSE)": {
        "HSBC Holdings": "HSBA.L",
        "BP": "BP.L",
        "GlaxoSmithKline": "GSK.L",
        "Barclays": "BARC.L",
        "Vodafone": "VOD.L"
    },
    "Germany (FWB)": {
        "SAP": "SAP.DE",
        "Siemens": "SIE.DE",
        "Volkswagen": "VOW3.DE"
    },
    "Japan (TSE)": {
        "Toyota": "7203.T",
        "Sony": "6758.T",
        "SoftBank": "9984.T"
    },
    "South Korea (KRX)": {
        "Samsung": "005930.KQ",
        "Hyundai": "005380.KQ",
        "SK Hynix": "000660.KQ"
    },
    "Brazil (B3)": {
        "Vale": "VALE3.SA",
        "Petrobras": "PETR4.SA",
        "Itaú Unibanco": "ITUB4.SA"
    },
    "Russia (MOEX)": {
        "Gazprom": "GAZP.ME",
        "Sberbank": "SBER.ME",
        "Lukoil": "LKOH.ME"
    },
    "South Africa (JSE)": {
        "Sasol": "SOL.JO",
        "Naspers": "NPN.JO",
        "Standard Bank": "SBK.JO"
    },
    "Switzerland (SIX)": {
        "Nestlé": "NESN.SW",
        "Roche": "ROG.SW",
        "Novartis": "NOVN.SW"
    },
    "Sweden (Nasdaq Stockholm)": {
        "Ericsson": "ERIC-B.ST",
        "Volvo": "VOLV-B.ST",
        "H&M": "HM-B.ST"
    },
    "Singapore (SGX)": {
        "SingTel": "Z74.SI",
        "DBS": "D05.SI",
        "Keppel Corp": "BN4.SI"
    },
    "Mexico (BMV)": {
        "América Móvil": "AMXL.MX",
        "Cemex": "CEMEXCPO.MX",
        "Grupo Bimbo": "BIMBOA.MX"
    },
    "Italy (Borsa Italiana)": {
        "Enel": "ENEL.MI",
        "Eni": "ENI.MI",
        "UniCredit": "UCG.MI"
    },
    "Netherlands (Euronext Amsterdam)": {
        "ASML": "ASML.AS",
        "Philips": "PHIA.AS",
        "Heineken": "HEIA.AS"
    }
}


# --- Header ---
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://img.icons8.com/fluency/96/stock-share.png", width=80)
with col2:
    st.title("StockSense")
    st.caption("Your AI-powered stock forecasting companion")

# --- Introduction Expander ---
with st.expander("ℹ️ About StockSense & Supported Markets", expanded=False):
    st.markdown("""
    Welcome to <b>StockSense</b> — an interactive platform to forecast global stock prices using a simplified AI model.
    Pick a market and company from the sidebar to see historical trends and generate short-term predictions.
    
    *This app is for educational purposes only and should not be used for financial advice.*
    """, unsafe_allow_html=True)
    # You can list the countries here if you wish

st.markdown("---")


# --- Sidebar Configuration ---
st.sidebar.header("📊 Configuration")
selected_country = st.sidebar.selectbox("Select Country/Market", list(top_companies_by_country.keys()))

company_map = top_companies_by_country[selected_country]
selected_company = st.sidebar.selectbox("Select Company", list(company_map.keys()))
ticker = company_map[selected_company]
days = st.sidebar.slider("Days to Predict", 1, 10, 5, help="Select the number of days into the future to forecast.")

# --- Main App Logic ---
# Data fetching and modeling now run automatically when a sidebar control is changed
st.header(f"🔮 Forecast for {selected_company}")

@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_data(ticker_symbol):
    return yf.download(ticker_symbol, period="3y", interval="1d")

data = load_data(ticker)

if data.empty:
    st.error("No data found for this company. Please try another.")
else:
    # --- Train Model ---
    df = data[['Close']].copy()
    df['Day'] = np.arange(len(df))
    X = df[['Day']]
    y = df['Close']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # --- Predict Future ---
    last_day = df['Day'].iloc[-1]
    future_days = np.arange(last_day + 1, last_day + 1 + days).reshape(-1, 1)
    preds = model.predict(future_days)

    future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, days + 1)]
    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': preds})
    
    # Prepare data for combined chart
    hist_df = data.reset_index()[['Date', 'Close']]
    hist_df['Type'] = 'Historical'
    pred_df_chart = pred_df.rename(columns={'Predicted Close': 'Close'})
    pred_df_chart['Type'] = 'Predicted'
    
    combined_df = pd.concat([hist_df, pred_df_chart])

    # --- Create Tabs for Results ---
    tab1, tab2, tab3 = st.tabs(["📈 Forecast Chart", "🔢 Prediction Data", "🧠 Model Info"])

    with tab1:
        st.subheader("Combined Price Chart")
        # Create a layered chart
        chart = alt.Chart(combined_df).mark_line().encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Close:Q', title='Close Price (USD)'),
            color=alt.Color('Type:N', scale=alt.Scale(domain=['Historical', 'Predicted'], range=['#1f77b4', '#ff7f0e'])),
            tooltip=['Date:T', 'Close:Q']
        ).properties(
            height=400
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

    with tab2:
        st.subheader(f"Predicted Prices for the Next {days} Days")
        st.dataframe(pred_df.style.format({"Predicted Close": "${:.2f}"}), use_container_width=True)
        
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Predictions",
            data=csv,
            file_name=f"{selected_company}_predictions.csv",
            mime="text/csv",
        )

    with tab3:
        st.subheader("About the Forecasting Model")
        st.warning("**Disclaimer:** This model is for educational purposes only.", icon="⚠️")
        st.markdown("""
        The predictions here are generated by a **Random Forest Regressor** model.
        - **How it works:** It's trained on the historical relationship between the day number (e.g., day 1, day 2,...) and the closing price.
        - **Limitations:** This is a simplistic model that acts as a **curve-fitter**. It does **not** perform a true time-series analysis and is unaware of market trends, volatility, or external news.
        - **Conclusion:** The forecast should be seen as a mathematical extension of the historical price curve, not as financial advice.
        """)

