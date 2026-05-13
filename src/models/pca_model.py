import numpy as np
from sklearn.decomposition import PCA


class PCAModel:
    def __init__(self, contamination=0.03):
        self.contamination = contamination
        # Reducing variance explained to 80% to make reconstruction more sensitive to outliers
        self.model = PCA(n_components=0.80, random_state=42)
        self.name = "PCA Reconstruction"

    def fit(self, X, y=None):
        self.model.fit(X)
        # Calculate threshold on training data
        X_pca = self.model.transform(X)
        X_reconstructed = self.model.inverse_transform(X_pca)
        train_errors = np.mean((X - X_reconstructed) ** 2, axis=1)
        self.threshold = np.quantile(train_errors, 1 - self.contamination)
        return self

    def predict(self, X):
        X_pca = self.model.transform(X)
        X_reconstructed = self.model.inverse_transform(X_pca)
        errors = np.mean((X - X_reconstructed) ** 2, axis=1)

        # Use stored threshold if available, otherwise calculate on the fly
        threshold = getattr(self, "threshold", np.quantile(errors, 1 - self.contamination))
        y_pred = (errors > threshold).astype(int)
        return y_pred, errors

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)