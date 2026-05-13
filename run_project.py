import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data_loader import download_data, save_raw_data
from src.preprocessing import preprocess_data, scale_features
from src.feature_engineering import create_features, get_feature_columns
from src.anomaly_injection import inject_synthetic_anomalies
from src.evaluation import evaluate_predictions, metrics_to_dataframe
from src.models.model_factory import get_model


def run_pipeline(ticker, start_date, end_date, selected_models, contamination=0.03):
    raw_path = f"data/raw/{ticker}_raw.csv"
    processed_path = f"data/processed/{ticker}_features.csv"

    df = download_data(ticker, start_date, end_date)
    save_raw_data(df, raw_path)

    df = preprocess_data(df)
    df, _ = inject_synthetic_anomalies(df, contamination=contamination)
    df = create_features(df)

    feature_cols = get_feature_columns()
    df_scaled, _ = scale_features(df, feature_cols)

    os.makedirs("data/processed", exist_ok=True)
    df_scaled.to_csv(processed_path, index=False)

    X = df_scaled[feature_cols].values
    y = df_scaled["label"].values

    # Train/Test Split to avoid overfitting in reported metrics
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, range(len(X)), test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    all_metrics = {}
    all_predictions = {}

    for model_name in selected_models:
        model = get_model(model_name, contamination=contamination)
        
        # Proper Evaluation: Fit on train, predict on test
        model.fit(X_train, y_train)
        y_pred_test, scores_test = model.predict(X_test)
        
        metrics, cm = evaluate_predictions(y_test, y_pred_test, scores_test)
        all_metrics[model_name] = metrics

        # Full Prediction for Visualization
        y_pred_full, scores_full = model.fit_predict(X, y)
        
        pred_df = df_scaled.copy()
        pred_df["prediction"] = y_pred_full
        pred_df["score"] = scores_full
        all_predictions[model_name] = {
            "data": pred_df,
            "confusion_matrix": cm,
            "metrics": metrics,
        }

    metrics_df = metrics_to_dataframe(all_metrics)

    os.makedirs("outputs/metrics", exist_ok=True)
    metrics_df.to_csv("outputs/metrics/comparison_metrics.csv", index=False)

    return metrics_df, all_predictions, X