import yfinance as yf

def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty:
            return None
        return hist['Close']
    except Exception as e:
        return None
