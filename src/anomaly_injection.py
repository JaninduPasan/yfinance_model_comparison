import numpy as np
import pandas as pd


def inject_synthetic_anomalies(df: pd.DataFrame, contamination: float = 0.03, random_state: int = 42):
    df = df.copy()
    rng = np.random.default_rng(random_state)

    n = len(df)
    anomaly_count = max(1, int(n * contamination))
    anomaly_indices = rng.choice(n, size=anomaly_count, replace=False)

    df["label"] = 0

    for idx in anomaly_indices:
        df.loc[idx, "Close"] *= rng.uniform(1.08, 1.20)
        df.loc[idx, "Volume"] *= rng.uniform(2.0, 5.0)
        df.loc[idx, "label"] = 1

    return df, anomaly_indices.tolist()