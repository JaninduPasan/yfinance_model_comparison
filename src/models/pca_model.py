import numpy as np
from sklearn.decomposition import PCA


class PCAModel:
    def __init__(self, contamination=0.03):
        self.contamination = contamination
        self.model = PCA(n_components=0.95)
        self.name = "PCA Reconstruction"

    def fit_predict(self, X, y=None):
        X_pca = self.model.fit_transform(X)
        X_reconstructed = self.model.inverse_transform(X_pca)
        errors = np.mean((X - X_reconstructed) ** 2, axis=1)

        threshold = np.quantile(errors, 1 - self.contamination)
        y_pred = (errors > threshold).astype(int)
        return y_pred, errors