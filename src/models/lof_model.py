from sklearn.neighbors import LocalOutlierFactor


class LOFModel:
    def __init__(self, contamination=0.03):
        self.model = LocalOutlierFactor(contamination=contamination, novelty=False)
        self.name = "Local Outlier Factor"

    def fit_predict(self, X, y=None):
        raw_pred = self.model.fit_predict(X)
        y_pred = (raw_pred == -1).astype(int)
        scores = -self.model.negative_outlier_factor_
        return y_pred, scores