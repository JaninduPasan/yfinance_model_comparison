import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop_duplicates()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date")
    df = df.fillna(method="ffill").fillna(method="bfill")

    return df


def scale_features(df: pd.DataFrame, feature_cols):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    scaled_df = df.copy()
    scaled_df[feature_cols] = scaled
    return scaled_df, scaler