import numpy as np
from sklearn.cluster import KMeans


class KMeansModel:
    def __init__(self, contamination=0.03):
        self.contamination = contamination
        self.model = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.name = "KMeans Distance"

    def fit_predict(self, X, y=None):
        self.model.fit(X)
        centers = self.model.cluster_centers_
        labels = self.model.labels_
        distances = np.linalg.norm(X - centers[labels], axis=1)

        threshold = np.quantile(distances, 1 - self.contamination)
        y_pred = (distances > threshold).astype(int)
        return y_pred, distances