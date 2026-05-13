import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestModel:
    def __init__(self, contamination=0.03):
        self.model = IsolationForest(
            n_estimators=200,
            max_samples='auto',
            contamination=contamination,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        self.name = "Isolation Forest"

    def fit(self, X, y=None):
        self.model.fit(X)
        return self

    def predict(self, X):
        raw_pred = self.model.predict(X)
        y_pred = np.where(raw_pred == -1, 1, 0)
        scores = -self.model.decision_function(X)
        return y_pred, scores

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)