import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.models.isolation_forest_model import IsolationForestModel
from src.models.one_class_svm_model import OneClassSVMModel
from src.models.lof_model import LOFModel
from src.models.elliptic_envelope_model import EllipticEnvelopeModel
from src.models.pca_model import PCAModel
from src.models.kmeans_model import KMeansModel
from src.models.autoencoder_model import AutoencoderModel
from src.models.random_forest_model import RandomForestModel
from src.models.gradient_boosting_model import GradientBoostingModel


class VotingEnsembleModel:
    def __init__(self, contamination=0.03):
        self.contamination = contamination
        self.name = "Voting Ensemble"

        self.base_models = [
            IsolationForestModel(contamination),
            OneClassSVMModel(contamination),
            LOFModel(contamination),
            PCAModel(contamination),
            AutoencoderModel(contamination),
        ]

    def fit(self, X, y=None):
        for model in self.base_models:
            if hasattr(model, "fit"):
                model.fit(X, y)
        return self

    def predict(self, X):
        predictions = []
        scores = []

        for model in self.base_models:
            if hasattr(model, "predict"):
                pred, score = model.predict(X)
            else:
                pred, score = model.fit_predict(X)
            predictions.append(pred)
            scores.append(score)

        pred_matrix = np.column_stack(predictions)
        score_matrix = np.column_stack(scores)

        # majority voting
        y_pred = (pred_matrix.sum(axis=1) >= (len(self.base_models) // 2 + 1)).astype(int)

        # average raw score
        avg_score = score_matrix.mean(axis=1)
        return y_pred, avg_score

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)


class AverageScoreEnsembleModel:
    def __init__(self, contamination=0.03):
        self.contamination = contamination
        self.name = "Average Score Ensemble"

        self.base_models = [
            IsolationForestModel(contamination),
            LOFModel(contamination),
            PCAModel(contamination),
            KMeansModel(contamination),
            AutoencoderModel(contamination),
        ]

    def fit(self, X, y=None):
        for model in self.base_models:
            if hasattr(model, "fit"):
                model.fit(X, y)
        return self

    def predict(self, X):
        scores = []

        for model in self.base_models:
            if hasattr(model, "predict"):
                _, score = model.predict(X)
            else:
                _, score = model.fit_predict(X)
            score = np.asarray(score).reshape(-1, 1)
            score = MinMaxScaler().fit_transform(score).flatten()
            scores.append(score)

        score_matrix = np.column_stack(scores)
        avg_score = score_matrix.mean(axis=1)

        threshold = np.quantile(avg_score, 1 - self.contamination)
        y_pred = (avg_score > threshold).astype(int)

        return y_pred, avg_score

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)


class WeightedHybridModel:
    def __init__(self, contamination=0.03):
        self.contamination = contamination
        self.name = "Weighted Hybrid Ensemble"

        self.unsupervised_models = [
            ("Isolation Forest", IsolationForestModel(contamination), 0.30),
            ("PCA Reconstruction", PCAModel(contamination), 0.20),
            ("Autoencoder", AutoencoderModel(contamination), 0.25),
            ("KMeans Distance", KMeansModel(contamination), 0.10),
            ("LOF", LOFModel(contamination), 0.15),
        ]

        self.supervised_models = [
            ("Random Forest", RandomForestModel(contamination), 0.60),
            ("Gradient Boosting", GradientBoostingModel(contamination), 0.40),
        ]

    def fit(self, X, y=None):
        for _, model, _ in self.unsupervised_models:
            if hasattr(model, "fit"):
                model.fit(X, y)
        if y is not None:
            for _, model, _ in self.supervised_models:
                if hasattr(model, "fit"):
                    model.fit(X, y)
        return self

    def predict(self, X):
        weighted_scores = []

        # unsupervised part
        for _, model, weight in self.unsupervised_models:
            if hasattr(model, "predict"):
                _, score = model.predict(X)
            else:
                _, score = model.fit_predict(X)
            score = np.asarray(score).reshape(-1, 1)
            score = MinMaxScaler().fit_transform(score).flatten()
            weighted_scores.append(score * weight)

        # supervised part
        for _, model, weight in self.supervised_models:
            try:
                if hasattr(model, "predict"):
                    _, score = model.predict(X)
                else:
                    _, score = model.fit_predict(X)
                score = np.asarray(score).reshape(-1, 1)
                score = MinMaxScaler().fit_transform(score).flatten()
                weighted_scores.append(score * weight)
            except Exception:
                continue

        final_score = np.sum(np.column_stack(weighted_scores), axis=1)

        threshold = np.quantile(final_score, 1 - self.contamination)
        y_pred = (final_score > threshold).astype(int)

        return y_pred, final_score

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)