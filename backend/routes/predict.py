from fastapi import APIRouter, HTTPException
from backend.schemas.request_response import PredictionRequest, PredictionResponse
from backend.services.prediction_service import PredictionService

router = APIRouter()
prediction_service = PredictionService()

@router.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest) -> PredictionResponse:
    """
    Predict sentiment of a given text.
    
    Args:
        request: PredictionRequest containing the text to analyze
        
    Returns:
        PredictionResponse with sentiment, confidence, and metadata
    """
    try:
        result = prediction_service.predict(
            text=request.text,
            model_type=request.model_type
        )
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@router.post("/batch-predict")
async def batch_predict(texts: list[str], model_type: str = "logistic_regression"):
    """
    Predict sentiment for multiple texts.
    
    Args:
        texts: List of texts to analyze
        model_type: Type of model to use
        
    Returns:
        List of predictions
    """
    try:
        results = [
            prediction_service.predict(text=text, model_type=model_type)
            for text in texts
        ]
        return {
            "count": len(results),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )

@router.get("/models")
async def get_available_models():
    """Get list of available models."""
    return {
        "models": [
            "logistic_regression",
            "svm",
            "bert",
            "custom_model"
        ]
    }
