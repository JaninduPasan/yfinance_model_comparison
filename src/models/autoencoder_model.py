import numpy as np
from sklearn.neural_network import MLPRegressor


class AutoencoderModel:
    def __init__(self, contamination=0.03):
        self.contamination = contamination
        self.model = None
        self.name = "Autoencoder"

    def fit_predict(self, X, y=None):
        input_dim = X.shape[1]
        hidden_dim = max(2, input_dim // 2)

        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_dim,),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42
        )

        self.model.fit(X, X)
        X_reconstructed = self.model.predict(X)
        errors = np.mean((X - X_reconstructed) ** 2, axis=1)

        threshold = np.quantile(errors, 1 - self.contamination)
        y_pred = (errors > threshold).astype(int)
        return y_pred, errors