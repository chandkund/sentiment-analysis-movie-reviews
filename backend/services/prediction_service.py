import time
from typing import Dict, Any
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionService:
    """Service for sentiment prediction using trained ML models."""
    
    def __init__(self):
        """Initialize prediction service with trained models."""
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self._load_models()
    
    def _load_models(self):
        """Load trained models from disk."""
        try:
            model_path = "models/model.pkl"
            vectorizer_path = "models/vectorizer.pkl"
            encoder_path = "models/label_encoder.pkl"
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Trained model loaded successfully")
            else:
                logger.warning("Trained model not found, using fallback")
                self._initialize_fallback_model()
            
            if os.path.exists(vectorizer_path):
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                logger.info("Vectorizer loaded successfully")
            else:
                logger.warning("Vectorizer not found, using fallback")
                self._initialize_fallback_vectorizer()
            
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info("Label encoder loaded successfully")
            else:
                logger.warning("Label encoder not found, using fallback")
                self._initialize_fallback_encoder()
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self._initialize_fallback_model()
    
    def _initialize_fallback_model(self):
        """Initialize fallback model for development."""
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression()
        logger.info("Fallback model initialized")
    
    def _initialize_fallback_vectorizer(self):
        """Initialize fallback vectorizer for development."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000)
        logger.info("Fallback vectorizer initialized")
    
    def _initialize_fallback_encoder(self):
        """Initialize fallback label encoder for development."""
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['negative', 'positive'])
        logger.info("Fallback label encoder initialized")
    
    def predict(self, text: str, model_type: str = "logistic_regression") -> Dict[str, Any]:
        """
        Predict sentiment of input text.
        
        Args:
            text: Input text to analyze
            model_type: Type of model to use
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        try:
            # Vectorize input text
            X = self.vectorizer.transform([text])
            
            # Get prediction and probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[0]
                prediction_numeric = self.model.predict(X)[0]
            else:
                prediction_numeric = self.model.predict(X)[0]
                probabilities = [0.0] * 2  # Binary classification
            
            # Convert numeric prediction back to string label
            if self.label_encoder:
                prediction_label = self.label_encoder.inverse_transform([prediction_numeric])[0]
            else:
                prediction_label = "positive" if prediction_numeric == 1 else "negative"
            
            # Map to sentiment
            sentiment = prediction_label
            
            # Calculate confidence
            confidence = float(max(probabilities)) if probabilities is not None else 0.8
            
            # Processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            return {
                "text": text,
                "sentiment": sentiment,
                "confidence": confidence,
                "scores": {
                    "negative": float(probabilities[0]) if len(probabilities) > 0 else 0.1,
                    "positive": float(probabilities[1]) if len(probabilities) > 1 else 0.1,
                    "neutral": 0.8 if sentiment == "neutral" else 0.1
                },
                "model_type": model_type,
                "processing_time_ms": processing_time_ms,
            }
        
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            return {
                "text": text,
                "sentiment": "neutral",
                "confidence": 0.0,
                "scores": {"negative": 0.0, "positive": 0.0, "neutral": 1.0},
                "model_type": model_type,
                "processing_time_ms": processing_time_ms,
                "error": str(e),
            }
    
    def batch_predict(self, texts: list[str], model_type: str = "logistic_regression") -> list[Dict[str, Any]]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of texts to analyze
            model_type: Type of model to use
            
        Returns:
            List of prediction results
        """
        return [self.predict(text, model_type) for text in texts]
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about available models."""
        return {
            "current_models": [
                "logistic_regression",
                "svm",
                "bert",
                "custom_model"
            ],
            "default_model": "logistic_regression",
            "version": "1.0.0",
        }
