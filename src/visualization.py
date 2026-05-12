import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def plot_price_with_anomalies(df: pd.DataFrame, pred_col: str, title: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Close"], label="Close Price")
    anomalies = df[df[pred_col] == 1]
    ax.scatter(anomalies["Date"], anomalies["Close"], color="red", label="Predicted Anomaly", s=25)
    ax.set_title(title)
    ax.legend()
    return fig


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    return fig


def plot_roc_curve(y_true, scores, title="ROC Curve"):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, scores, ax=ax)
    ax.set_title(title)
    return fig


def plot_pr_curve(y_true, scores, title="Precision-Recall Curve"):
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, scores, ax=ax)
    ax.set_title(title)
    return fig

def plot_clustering_view(X, y_pred=None, n_clusters=3, title="Clustering View"):
    """
    Reduce features to 2D using PCA and show clusters/anomalies.
    X: scaled feature matrix
    y_pred: predicted anomaly labels (0 = normal, 1 = anomaly)
    """
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_2d)

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=cluster_labels,
        cmap="tab10",
        alpha=0.7,
        s=35,
        label="Data Points"
    )

    if y_pred is not None:
        anomalies = np.where(y_pred == 1)[0]
        if len(anomalies) > 0:
           ax.scatter(
                X_2d[anomalies, 0],
                X_2d[anomalies, 1],
                color="red",
                edgecolors="black",
                s=80,
                label="Predicted Anomalies"
           )

    ax.set_title(title)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend()
    return fig