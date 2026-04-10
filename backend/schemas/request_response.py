from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class PredictionRequest(BaseModel):
    """Request schema for sentiment prediction."""
    
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to analyze for sentiment"
    )
    model_type: str = Field(
        default="logistic_regression",
        description="Type of ML model to use for prediction"
    )
    return_probabilities: bool = Field(
        default=True,
        description="Whether to return probability scores"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This movie was absolutely amazing! Best film of the year.",
                "model_type": "logistic_regression",
                "return_probabilities": True
            }
        }


class SentimentScore(BaseModel):
    """Sentiment score breakdown."""
    
    positive: float = Field(..., ge=0, le=1, description="Positive sentiment probability")
    negative: float = Field(..., ge=0, le=1, description="Negative sentiment probability")
    neutral: float = Field(..., ge=0, le=1, description="Neutral sentiment probability")


class PredictionResponse(BaseModel):
    """Response schema for sentiment prediction."""
    
    text: str = Field(..., description="Original input text")
    sentiment: str = Field(..., description="Predicted sentiment: positive, negative, or neutral")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score of the prediction")
    scores: Optional[SentimentScore] = Field(
        default=None,
        description="Detailed probability scores per sentiment"
    )
    model_type: str = Field(..., description="Model used for prediction")
    processing_time_ms: float = Field(..., description="Time taken for prediction in milliseconds")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This movie was absolutely amazing!",
                "sentiment": "positive",
                "confidence": 0.95,
                "scores": {
                    "positive": 0.95,
                    "negative": 0.03,
                    "neutral": 0.02
                },
                "model_type": "logistic_regression",
                "processing_time_ms": 12.5
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    
    count: int = Field(..., description="Number of predictions made")
    predictions: list[PredictionResponse] = Field(..., description="List of predictions")
