from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    def __init__(self, contamination=0.03):
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.name = "Random Forest"

    def fit_predict(self, X, y=None):
        if y is None:
            raise ValueError("Supervised model requires labels.")
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        scores = self.model.predict_proba(X)[:, 1]
        return y_pred, scores