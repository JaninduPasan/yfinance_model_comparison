import numpy as np
from sklearn.covariance import EllipticEnvelope


class EllipticEnvelopeModel:
    def __init__(self, contamination=0.03):
        self.model = EllipticEnvelope(contamination=contamination, random_state=42)
        self.name = "Elliptic Envelope"

    def fit_predict(self, X, y=None):
        self.model.fit(X)
        raw_pred = self.model.predict(X)
        y_pred = np.where(raw_pred == -1, 1, 0)
        scores = -self.model.decision_function(X)
        return y_pred, scores