import streamlit as st
import pandas as pd

from run_project import run_pipeline
from src.visualization import (
    plot_price_with_anomalies,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_clustering_view
)
from src.models.model_factory import MODEL_REGISTRY

st.set_page_config(page_title="Stock Anomaly Detection", layout="wide")

st.title("Explainable Stock Market Anomaly Detection")
st.write("Compare multiple anomaly detection and classification models on yfinance stock data.")

with st.sidebar:
    st.header("Configuration")

    ticker_dict = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Google": "GOOGL",
        "Amazon": "AMZN",
        "Tesla": "TSLA",
        "Nvidia": "NVDA",
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD"
    }

    ticker_name = st.selectbox("Asset", list(ticker_dict.keys()))
    ticker = ticker_dict[ticker_name]

    start_date = st.date_input("Start Date", value=pd.to_datetime("2019-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2025-01-01"))
    contamination = st.slider("Synthetic anomaly ratio", 0.001, 0.10, 0.03, 0.001)
    n_clusters = st.slider("Number of clusters", 2, 8, 3)

    selected_models = st.multiselect(
        "Select Models",
        options=list(MODEL_REGISTRY.keys()),
        default=["Isolation Forest", "Random Forest", "Autoencoder"]
    )

    run_btn = st.button("Run Comparison")

if run_btn:
    if not selected_models:
        st.warning("Select at least one model.")
    else:
        with st.spinner("Running models..."):
            metrics_df, predictions, X = run_pipeline(
                ticker=ticker,
                start_date=str(start_date),
                end_date=str(end_date),
                selected_models=selected_models,
                contamination=contamination
            )

        st.subheader("Model Comparison Metrics")
        st.dataframe(metrics_df, use_container_width=True)

        st.subheader("Comparison Chart")
        st.bar_chart(metrics_df.set_index("model")[["accuracy", "precision", "recall", "f1_score"]])

        for model_name in selected_models:
            st.markdown(f"---")
            st.header(model_name)

            model_result = predictions[model_name]
            pred_df = model_result["data"]
            cm = model_result["confusion_matrix"]

            col1, col2 = st.columns(2)

            with col1:
                st.pyplot(plot_price_with_anomalies(pred_df, "prediction", f"{model_name} - Price with Predicted Anomalies"))

            with col2:
                st.pyplot(plot_confusion_matrix(cm, f"{model_name} - Confusion Matrix"))

            if pred_df["label"].nunique() > 1:
                col3, col4 = st.columns(2)

                with col3:
                    st.pyplot(plot_roc_curve(pred_df["label"], pred_df["score"], f"{model_name} - ROC Curve"))

                with col4:
                    st.pyplot(plot_pr_curve(pred_df["label"], pred_df["score"], f"{model_name} - Precision-Recall Curve"))

            st.subheader(f"{model_name} - Clustering View")
            st.pyplot(plot_clustering_view(
                X,
                y_pred=pred_df["prediction"].values,
                n_clusters=n_clusters,
                title=f"{model_name} - PCA Clustering with Anomalies"
            ))

            # st.subheader(f"{model_name} Predictions")
            # st.dataframe(
            #     pred_df[["Date", "Close", "Volume", "label", "prediction", "score"]].head(50),
            #     use_container_width=True
            # )
