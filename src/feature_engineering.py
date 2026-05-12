import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["price_change"] = df["Close"].pct_change()
    df["high_low_spread"] = (df["High"] - df["Low"]) / df["Close"]
    df["open_close_spread"] = (df["Open"] - df["Close"]) / df["Close"]

    df["rolling_mean_5"] = df["Close"].rolling(5).mean()
    df["rolling_mean_10"] = df["Close"].rolling(10).mean()
    df["rolling_std_5"] = df["Close"].rolling(5).std()
    df["rolling_std_10"] = df["Close"].rolling(10).std()

    df["volume_change"] = df["Volume"].pct_change()
    df["volume_ma_5"] = df["Volume"].rolling(5).mean()
    df["volume_spike"] = df["Volume"] / df["volume_ma_5"]

    df["momentum_5"] = df["Close"] - df["Close"].shift(5)
    df["momentum_10"] = df["Close"] - df["Close"].shift(10)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    return df


def get_feature_columns():
    return [
        "return",
        "price_change",
        "high_low_spread",
        "open_close_spread",
        "rolling_mean_5",
        "rolling_mean_10",
        "rolling_std_5",
        "rolling_std_10",
        "volume_change",
        "volume_ma_5",
        "volume_spike",
        "momentum_5",
        "momentum_10",
    ]