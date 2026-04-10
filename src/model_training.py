from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

def get_models():
    """Get dictionary of ML models for training."""
    return {
        "logistic": LogisticRegression(),
        "svm": LinearSVC(),
        "rf": RandomForestClassifier()
    }