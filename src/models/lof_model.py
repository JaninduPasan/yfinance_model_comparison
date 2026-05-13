from sklearn.neighbors import LocalOutlierFactor


class LOFModel:
    def __init__(self, contamination=0.03):
        self.model = LocalOutlierFactor(
            n_neighbors=35,
            contamination=contamination,
            novelty=True
        )
        self.name = "Local Outlier Factor"

    def fit(self, X, y=None):
        self.model.fit(X)
        return self

    def predict(self, X):
        raw_pred = self.model.predict(X)
        y_pred = (raw_pred == -1).astype(int)
        # For scores, we use decision_function when novelty=True
        scores = -self.model.decision_function(X)
        return y_pred, scores

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)