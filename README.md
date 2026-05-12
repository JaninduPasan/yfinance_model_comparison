# yfinance_anomaly_project

A Streamlit-based stock market anomaly detection project using Yahoo Finance data.

## Features
- Download OHLCV data from yfinance
- Preprocess and engineer financial features
- Inject synthetic anomalies for supervised comparison
- Compare multiple models
- Evaluate using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - MCC
  - ROC-AUC
  - PR-AUC
- Visualize results in Streamlit UI

## Run

```bash
pip install -r requirements.txt
streamlit run app.py