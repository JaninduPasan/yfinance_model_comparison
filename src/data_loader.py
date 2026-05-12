import os
import pandas as pd
import yfinance as yf


def download_data(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=False)

    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    return df


def save_raw_data(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)