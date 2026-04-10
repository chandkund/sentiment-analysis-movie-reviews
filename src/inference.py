import numpy as np
import pickle
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceEngine:
    """Make predictions using trained models."""
    
    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model
            vectorizer_path: Path to fitted vectorizer
        """
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        
        if model_path:
            self.load_model(model_path)
        if vectorizer_path:
            self.load_vectorizer(vectorizer_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to model file
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_vectorizer(self, vectorizer_path: str) -> None:
        """
        Load fitted vectorizer from disk.
        
        Args:
            vectorizer_path: Path to vectorizer file
        """
        try:
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info(f"Vectorizer loaded from {vectorizer_path}")
        except Exception as e:
            logger.error(f"Error loading vectorizer: {e}")
            raise
    
    def set_model(self, model: Any) -> None:
        """
        Set model programmatically.
        
        Args:
            model: Trained model object
        """
        self.model = model
        logger.info("Model set in inference engine")
    
    def set_vectorizer(self, vectorizer: Any) -> None:
        """
        Set vectorizer programmatically.
        
        Args:
            vectorizer: Fitted vectorizer object
        """
        self.vectorizer = vectorizer
        logger.info("Vectorizer set in inference engine")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Make single prediction on input text.
        
        Args:
            text: Input text for prediction
            
        Returns:
            Prediction result with label and probabilities
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model or vectorizer not loaded")
        
        try:
            # Vectorize input
            X = self.vectorizer.transform([text])
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            # Get probabilities if available
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[0]
            elif hasattr(self.model, 'decision_function'):
                decision = self.model.decision_function(X)[0]
                # Convert to probabilities for binary classification
                if len(decision.shape) == 0:
                    probabilities = [1 - 1/(1 + np.exp(decision)), 1/(1 + np.exp(decision))]
                else:
                    probabilities = np.exp(decision) / np.exp(decision).sum()
            
            # Map prediction to sentiment label
            sentiment_map = {0: "negative", 1: "positive", 2: "neutral"}
            sentiment = sentiment_map.get(prediction, "neutral")
            
            result = {
                'text': text,
                'prediction': sentiment,
                'prediction_code': int(prediction),
                'confidence': float(max(probabilities)) if probabilities is not None else 0.0,
            }
            
            if probabilities is not None:
                result['probabilities'] = {
                    'negative': float(probabilities[0]),
                    'positive': float(probabilities[1]),
                    'neutral': float(probabilities[2]) if len(probabilities) > 2 else 0.0
                }
            
            return result
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Make predictions on multiple texts.
        
        Args:
            texts: List of texts for prediction
            
        Returns:
            List of prediction results
        """
        logger.info(f"Making batch predictions on {len(texts)} texts...")
        predictions = [self.predict(text) for text in texts]
        logger.info(f"Batch prediction completed")
        return predictions
    
    def predict_with_confidence_threshold(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Make prediction with confidence filtering.
        
        Args:
            text: Input text for prediction
            threshold: Confidence threshold (0-1)
            
        Returns:
            Prediction result with threshold check
        """
        result = self.predict(text)
        
        result['confidence_threshold'] = threshold
        result['meets_threshold'] = result['confidence'] >= threshold
        
        if not result['meets_threshold']:
            result['prediction'] = 'uncertain'
            logger.warning(f"Confidence {result['confidence']:.2f} below threshold {threshold}")
        
        return result
    
    def predict_top_n(self, text: str, n: int = 1) -> List[Dict[str, Any]]:
        """
        Get top N predictions with probabilities.
        
        Args:
            text: Input text for prediction
            n: Number of top predictions to return
            
        Returns:
            List of top N predictions with scores
        """
        result = self.predict(text)
        
        if 'probabilities' not in result:
            return [result]
        
        # Sort by confidence
        probs = result['probabilities']
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        top_predictions = []
        for i, (label, score) in enumerate(sorted_probs[:n]):
            top_predictions.append({
                'rank': i + 1,
                'sentiment': label,
                'confidence': float(score)
            })
        
        return top_predictions
    
    def explain_prediction(self, text: str) -> Dict[str, Any]:
        """
        Generate explanation for prediction.
        
        Args:
            text: Input text for prediction
            
        Returns:
            Prediction with explanation
        """
        result = self.predict(text)
        
        # Get feature importance if available
        explanation = {
            'prediction': result,
            'input_length': len(text),
            'word_count': len(text.split()),
        }
        
        # Add feature importance for tree-based models
        if hasattr(self.model, 'feature_importances_'):
            explanation['has_feature_importance'] = True
            explanation['top_features'] = "Model explanation not yet implemented"
        
        return explanation
