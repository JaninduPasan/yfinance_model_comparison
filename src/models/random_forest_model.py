from sklearn.ensemble import RandomForestClassifier
import numpy as np


class RandomForestModel:
    def __init__(self, contamination=0.03):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        self.name = "Random Forest"

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Supervised model requires labels.")
        if len(np.unique(y)) > 1:
            self.model.fit(X, y)
        return self

    def predict(self, X):
        try:
            y_pred = self.model.predict(X)
            scores = self.model.predict_proba(X)[:, 1]
        except Exception:
            # Handle case where model isn't fitted or only one class was present
            y_pred = np.zeros(len(X))
            scores = np.zeros(len(X))
        return y_pred, scores

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)