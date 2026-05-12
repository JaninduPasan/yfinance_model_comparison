from src.models.isolation_forest_model import IsolationForestModel
from src.models.one_class_svm_model import OneClassSVMModel
from src.models.lof_model import LOFModel
from src.models.elliptic_envelope_model import EllipticEnvelopeModel
from src.models.pca_model import PCAModel
from src.models.kmeans_model import KMeansModel
from src.models.autoencoder_model import AutoencoderModel
from src.models.logistic_regression_model import LogisticRegressionModel
from src.models.random_forest_model import RandomForestModel
from src.models.gradient_boosting_model import GradientBoostingModel
from src.models.svm_classifier_model import SVMClassifierModel
from src.models.combined_models import (
    VotingEnsembleModel,
    AverageScoreEnsembleModel,
    WeightedHybridModel
)

MODEL_REGISTRY = {
    "Isolation Forest": IsolationForestModel,
    "One-Class SVM": OneClassSVMModel,
    "Local Outlier Factor": LOFModel,
    "Elliptic Envelope": EllipticEnvelopeModel,
    "PCA Reconstruction": PCAModel,
    "KMeans Distance": KMeansModel,
    "Autoencoder": AutoencoderModel,
    "Logistic Regression": LogisticRegressionModel,
    "Random Forest": RandomForestModel,
    "Gradient Boosting": GradientBoostingModel,
    "SVM Classifier": SVMClassifierModel,

    # Combined models
    "Voting Ensemble": VotingEnsembleModel,
    "Average Score Ensemble": AverageScoreEnsembleModel,
    "Weighted Hybrid Ensemble": WeightedHybridModel,
}


def get_model(model_name, contamination=0.03):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}")
    return MODEL_REGISTRY[model_name](contamination=contamination)