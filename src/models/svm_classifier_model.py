import numpy as np
from sklearn.svm import SVC


class SVMClassifierModel:
    def __init__(self, contamination=0.03):
        self.model = SVC(
            probability=True,
            class_weight="balanced",
            kernel="rbf",
            random_state=42
        )
        self.name = "SVM Classifier"

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
            y_pred = np.zeros(len(X))
            scores = np.zeros(len(X))
        return y_pred, scores

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)