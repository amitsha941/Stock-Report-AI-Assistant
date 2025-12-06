# stock_tools.py
import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, period="6mo", interval="1d"):
    """Fetch historical stock data."""
    data = yf.download(ticker, period=period, interval=interval)
    return data

def compute_indicators(data: pd.DataFrame):
    """Compute basic moving averages and RSI."""
    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["MA50"] = data["Close"].rolling(window=50).mean()
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    return data

def stock_summary(data, ticker):
    """Return readable stock summary."""
    last = data.iloc[-1]
    return (
        f"📈 **{ticker} Stock Overview**\n"
        f"- Date: {data.index[-1].date()}\n"
        f"- Close: {last['Close']:.2f}\n"
        f"- MA20: {last['MA20']:.2f}, MA50: {last['MA50']:.2f}\n"
        f"- RSI: {last['RSI']:.2f}\n"
        f"- Volume: {int(last['Volume'])}\n"
    )
