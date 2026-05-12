from sklearn.svm import SVC


class SVMClassifierModel:
    def __init__(self, contamination=0.03):
        self.model = SVC(probability=True, random_state=42)
        self.name = "SVM Classifier"

    def fit_predict(self, X, y=None):
        if y is None:
            raise ValueError("Supervised model requires labels.")
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        scores = self.model.predict_proba(X)[:, 1]
        return y_pred, scores