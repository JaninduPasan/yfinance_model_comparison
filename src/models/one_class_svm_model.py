import numpy as np
from sklearn.svm import OneClassSVM


class OneClassSVMModel:
    def __init__(self, contamination=0.03):
        self.model = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
        self.name = "One-Class SVM"

    def fit_predict(self, X, y=None):
        self.model.fit(X)
        raw_pred = self.model.predict(X)
        y_pred = np.where(raw_pred == -1, 1, 0)
        scores = -self.model.decision_function(X)
        return y_pred, scores