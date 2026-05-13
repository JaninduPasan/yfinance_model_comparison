import numpy as np
from sklearn.neural_network import MLPRegressor


class AutoencoderModel:
    def __init__(self, contamination=0.03):
        self.contamination = contamination
        self.model = None
        self.name = "Autoencoder"

    def fit(self, X, y=None):
        if self.model is None:
            input_dim = X.shape[1]
            bottleneck_dim = max(2, input_dim // 3)
            hidden_layers = (max(32, input_dim * 2), bottleneck_dim, max(32, input_dim * 2))
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation="relu",
                solver="adam",
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
                alpha=0.1
            )
        self.model.fit(X, X)
        
        # Calculate threshold on training data
        X_reconstructed = self.model.predict(X)
        train_errors = np.mean((X - X_reconstructed) ** 2, axis=1)
        self.threshold = np.quantile(train_errors, 1 - self.contamination)
        return self

    def predict(self, X):
        X_reconstructed = self.model.predict(X)
        errors = np.mean((X - X_reconstructed) ** 2, axis=1)
        
        # Use stored threshold if available, otherwise calculate on the fly
        threshold = getattr(self, "threshold", np.quantile(errors, 1 - self.contamination))
        y_pred = (errors > threshold).astype(int)
        return y_pred, errors

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)