from sklearn.ensemble import GradientBoostingClassifier


class GradientBoostingModel:
    def __init__(self, contamination=0.03):
        self.model = GradientBoostingClassifier(random_state=42)
        self.name = "Gradient Boosting"

    def fit_predict(self, X, y=None):
        if y is None:
            raise ValueError("Supervised model requires labels.")
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        scores = self.model.predict_proba(X)[:, 1]
        return y_pred, scores