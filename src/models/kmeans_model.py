import numpy as np
from sklearn.cluster import KMeans


class KMeansModel:
    def __init__(self, contamination=0.03):
        self.contamination = contamination
        # Using more clusters to better capture the complex normal data manifold
        self.model = KMeans(n_clusters=8, random_state=42, n_init=10)
        self.name = "KMeans Distance"

    def fit(self, X, y=None):
        self.model.fit(X)
        # Calculate threshold on training data
        centers = self.model.cluster_centers_
        labels = self.model.labels_
        train_distances = np.linalg.norm(X - centers[labels], axis=1)
        self.threshold = np.quantile(train_distances, 1 - self.contamination)
        return self

    def predict(self, X):
        centers = self.model.cluster_centers_
        labels = self.model.predict(X)
        distances = np.linalg.norm(X - centers[labels], axis=1)

        # Use stored threshold if available, otherwise calculate on the fly
        threshold = getattr(self, "threshold", np.quantile(distances, 1 - self.contamination))
        y_pred = (distances > threshold).astype(int)
        return y_pred, distances

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)