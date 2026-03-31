"""
Sentiment Analysis API Routes
Provides endpoints for sentiment analysis functionality
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import json
from models.sentiment_model import SentimentModel
import io

router = APIRouter()

# Global model instance
sentiment_model = SentimentModel()

class SentimentRequest(BaseModel):
    texts: List[str]
    method: str = "ml_model"  # "ml_model" or "textblob"

class SentimentResponse(BaseModel):
    results: List[Dict[str, Any]]
    method: str
    total_processed: int

@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze sentiment for a list of texts
    
    Args:
        request: SentimentRequest containing texts and method
        
    Returns:
        SentimentResponse: Analysis results
    """
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided for analysis")
        
        if request.method == "ml_model":
            if not sentiment_model.is_trained:
                raise HTTPException(
                    status_code=400, 
                    detail="ML model not trained. Please train the model first."
                )
            results = sentiment_model.predict(request.texts)
        elif request.method == "textblob":
            results = [sentiment_model.get_textblob_sentiment(text) for text in request.texts]
        else:
            raise HTTPException(status_code=400, detail="Invalid method specified")
        
        return SentimentResponse(
            results=results,
            method=request.method,
            total_processed=len(results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

@router.post("/train")
async def train_sentiment_model(
    training_data: UploadFile = File(...),
    model_type: str = "naive_bayes",
    text_column: str = "text",
    label_column: str = "sentiment"
):
    """
    Train the sentiment analysis model
    
    Args:
        training_data: CSV file with training data
        model_type: Type of model to train
        text_column: Column name for text data
        label_column: Column name for sentiment labels
        
    Returns:
        Training results and metrics
    """
    try:
        # Read uploaded file
        contents = await training_data.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        if text_column not in df.columns or label_column not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Required columns '{text_column}' and/or '{label_column}' not found in data"
            )
        
        # Create new model instance
        global sentiment_model
        sentiment_model = SentimentModel(model_type=model_type)
        
        # Train model
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()
        
        training_results = sentiment_model.train(texts, labels)
        
        return {
            "message": "Model trained successfully",
            "model_type": model_type,
            "training_samples": len(texts),
            "metrics": training_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@router.get("/model/status")
async def get_model_status():
    """
    Get the current status of the sentiment model
    
    Returns:
        Model status information
    """
    return {
        "is_trained": sentiment_model.is_trained,
        "model_type": sentiment_model.model_type,
        "model_loaded": sentiment_model.model is not None
    }

@router.post("/model/save")
async def save_model(filename: str = "sentiment_model.joblib"):
    """
    Save the trained sentiment model
    
    Args:
        filename: Name for the saved model file
        
    Returns:
        Save confirmation
    """
    try:
        if not sentiment_model.is_trained:
            raise HTTPException(status_code=400, detail="No trained model to save")
        
        filepath = f"saved_models/{filename}"
        sentiment_model.save_model(filepath)
        
        return {
            "message": "Model saved successfully",
            "filepath": filepath
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving model: {str(e)}")

@router.post("/model/load")
async def load_model(
    model_file: UploadFile = File(...)
):
    """
    Load a trained sentiment model
    
    Args:
        model_file: Joblib file containing trained model
        
    Returns:
        Load confirmation
    """
    try:
        # Save uploaded file temporarily
        contents = await model_file.read()
        temp_filepath = f"temp_{model_file.filename}"
        
        with open(temp_filepath, 'wb') as f:
            f.write(contents)
        
        # Load model
        global sentiment_model
        sentiment_model = SentimentModel()
        sentiment_model.load_model(temp_filepath)
        
        # Clean up temporary file
        import os
        os.remove(temp_filepath)
        
        return {
            "message": "Model loaded successfully",
            "model_type": sentiment_model.model_type,
            "is_trained": sentiment_model.is_trained
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@router.post("/analyze/batch")
async def analyze_sentiment_batch(
    data_file: UploadFile = File(...),
    text_column: str = "text",
    method: str = "ml_model"
):
    """
    Analyze sentiment for texts in a CSV file
    
    Args:
        data_file: CSV file containing texts to analyze
        text_column: Column name containing text data
        method: Analysis method to use
        
    Returns:
        Batch analysis results
    """
    try:
        # Read uploaded file
        contents = await data_file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate text column
        if text_column not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Column '{text_column}' not found in data"
            )
        
        # Extract texts
        texts = df[text_column].dropna().tolist()
        
        if not texts:
            raise HTTPException(status_code=400, detail="No valid texts found in the specified column")
        
        # Analyze sentiment
        if method == "ml_model":
            if not sentiment_model.is_trained:
                raise HTTPException(
                    status_code=400, 
                    detail="ML model not trained. Please train the model first."
                )
            results = sentiment_model.predict(texts)
        elif method == "textblob":
            results = [sentiment_model.get_textblob_sentiment(text) for text in texts]
        else:
            raise HTTPException(status_code=400, detail="Invalid method specified")
        
        return {
            "message": "Batch analysis completed",
            "method": method,
            "total_processed": len(results),
            "results": results,
            "original_data_shape": df.shape
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in batch analysis: {str(e)}")

@router.get("/sentiment/distribution")
async def get_sentiment_distribution(results: List[Dict[str, Any]]):
    """
    Calculate sentiment distribution from analysis results
    
    Args:
        results: List of sentiment analysis results
        
    Returns:
        Sentiment distribution statistics
    """
    try:
        if not results:
            raise HTTPException(status_code=400, detail="No results provided")
        
        # Count sentiments
        sentiment_counts = {}
        confidence_scores = []
        
        for result in results:
            sentiment = result.get('sentiment_label', result.get('sentiment', 'unknown'))
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            if 'confidence' in result:
                confidence_scores.append(result['confidence'])
        
        total = len(results)
        sentiment_distribution = {
            sentiment: {
                'count': count,
                'percentage': (count / total) * 100
            }
            for sentiment, count in sentiment_counts.items()
        }
        
        response = {
            "total_analyzed": total,
            "sentiment_distribution": sentiment_distribution
        }
        
        if confidence_scores:
            response["confidence_stats"] = {
                "mean_confidence": sum(confidence_scores) / len(confidence_scores),
                "min_confidence": min(confidence_scores),
                "max_confidence": max(confidence_scores)
            }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating distribution: {str(e)}")
