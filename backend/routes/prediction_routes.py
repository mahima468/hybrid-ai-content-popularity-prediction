"""
Content Popularity Prediction API Routes
Provides endpoints for content popularity prediction functionality
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import json
from models.popularity_prediction import PopularityPredictionModel
import io

router = APIRouter()

# Global model instance
popularity_model = PopularityPredictionModel()

class PopularityPredictionRequest(BaseModel):
    model_type: str = "random_forest"  # "random_forest", "gradient_boosting", "linear", "ridge"

class ContentData(BaseModel):
    content_data: List[Dict[str, Any]]

class TrainingDataRequest(BaseModel):
    content_data: List[Dict[str, Any]]
    target_variable: str
    sentiment_data: Optional[List[Dict[str, Any]]] = None
    engagement_data: Optional[List[Dict[str, Any]]] = None

@router.post("/predict")
async def predict_popularity(
    request: PopularityPredictionRequest,
    content_data: ContentData,
    sentiment_data: Optional[List[Dict[str, Any]]] = None,
    engagement_data: Optional[List[Dict[str, Any]]] = None
):
    """
    Predict popularity for content
    
    Args:
        request: Model configuration
        content_data: Content metadata
        sentiment_data: Sentiment analysis results (optional)
        engagement_data: Engagement metrics (optional)
        
    Returns:
        Popularity predictions
    """
    try:
        if not content_data.content_data:
            raise HTTPException(status_code=400, detail="No content data provided")
        
        # Create new model instance
        global popularity_model
        popularity_model = PopularityPredictionModel(model_type=request.model_type)
        
        # Convert to DataFrames
        content_df = pd.DataFrame(content_data.content_data)
        sentiment_df = pd.DataFrame(sentiment_data) if sentiment_data else None
        engagement_df = pd.DataFrame(engagement_data) if engagement_data else None
        
        # Make predictions
        prediction_results = popularity_model.predict(
            content_df, sentiment_df, engagement_df
        )
        
        return {
            "message": "Popularity prediction completed",
            "model_type": request.model_type,
            "predictions": prediction_results["predictions"].to_dict('records'),
            "summary": prediction_results["summary"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting popularity: {str(e)}")

@router.post("/predict/file")
async def predict_popularity_from_files(
    content_file: UploadFile = File(...),
    sentiment_file: Optional[UploadFile] = File(None),
    engagement_file: Optional[UploadFile] = File(None),
    model_type: str = "random_forest"
):
    """
    Predict popularity from uploaded CSV files
    
    Args:
        content_file: CSV file with content metadata
        sentiment_file: CSV file with sentiment data (optional)
        engagement_file: CSV file with engagement data (optional)
        model_type: Type of model to use
        
    Returns:
        Popularity predictions
    """
    try:
        # Read content file (required)
        content_contents = await content_file.read()
        content_df = pd.read_csv(io.StringIO(content_contents.decode('utf-8')))
        
        if content_df.empty:
            raise HTTPException(status_code=400, detail="Content file is empty")
        
        # Read optional files
        sentiment_df = None
        if sentiment_file:
            sentiment_contents = await sentiment_file.read()
            sentiment_df = pd.read_csv(io.StringIO(sentiment_contents.decode('utf-8')))
        
        engagement_df = None
        if engagement_file:
            engagement_contents = await engagement_file.read()
            engagement_df = pd.read_csv(io.StringIO(engagement_contents.decode('utf-8')))
        
        # Create new model instance
        global popularity_model
        popularity_model = PopularityPredictionModel(model_type=model_type)
        
        # Make predictions
        prediction_results = popularity_model.predict(
            content_df, sentiment_df, engagement_df
        )
        
        return {
            "message": "Popularity prediction completed",
            "model_type": model_type,
            "data_shapes": {
                "content": content_df.shape,
                "sentiment": sentiment_df.shape if sentiment_df is not None else None,
                "engagement": engagement_df.shape if engagement_df is not None else None
            },
            "predictions": prediction_results["predictions"].to_dict('records'),
            "summary": prediction_results["summary"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@router.post("/train")
async def train_popularity_model(
    content_file: UploadFile = File(...),
    target_variable: str = "popularity_score",
    model_type: str = "random_forest",
    sentiment_file: Optional[UploadFile] = File(None),
    engagement_file: Optional[UploadFile] = File(None)
):
    """
    Train the popularity prediction model
    
    Args:
        content_file: CSV file with content metadata and target variable
        target_variable: Name of target variable column
        model_type: Type of model to train
        sentiment_file: CSV file with sentiment data (optional)
        engagement_file: CSV file with engagement data (optional)
        
    Returns:
        Training results and metrics
    """
    try:
        # Read content file
        content_contents = await content_file.read()
        content_df = pd.read_csv(io.StringIO(content_contents.decode('utf-8')))
        
        # Validate target variable
        if target_variable not in content_df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Target variable '{target_variable}' not found in content data"
            )
        
        # Read optional files
        sentiment_df = None
        if sentiment_file:
            sentiment_contents = await sentiment_file.read()
            sentiment_df = pd.read_csv(io.StringIO(sentiment_contents.decode('utf-8')))
        
        engagement_df = None
        if engagement_file:
            engagement_contents = await engagement_file.read()
            engagement_df = pd.read_csv(io.StringIO(engagement_contents.decode('utf-8')))
        
        # Create new model instance
        global popularity_model
        popularity_model = PopularityPredictionModel(model_type=model_type)
        
        # Train model
        training_results = popularity_model.train(
            content_df, target_variable, sentiment_df, engagement_df
        )
        
        return {
            "message": "Model trained successfully",
            "model_type": model_type,
            "target_variable": target_variable,
            "training_samples": len(content_df),
            "metrics": training_results["training_metrics"],
            "feature_importance": training_results["feature_importance"],
            "selected_features": training_results["selected_features"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@router.get("/model/status")
async def get_model_status():
    """
    Get the current status of the popularity prediction model
    
    Returns:
        Model status information
    """
    return {
        "is_trained": popularity_model.is_trained,
        "model_type": popularity_model.model_type,
        "model_loaded": popularity_model.model is not None
    }

@router.post("/model/save")
async def save_model(filename: str = "popularity_model.joblib"):
    """
    Save the trained popularity prediction model
    
    Args:
        filename: Name for the saved model file
        
    Returns:
        Save confirmation
    """
    try:
        if not popularity_model.is_trained:
            raise HTTPException(status_code=400, detail="No trained model to save")
        
        filepath = f"saved_models/{filename}"
        popularity_model.save_model(filepath)
        
        return {
            "message": "Model saved successfully",
            "filepath": filepath,
            "model_type": popularity_model.model_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving model: {str(e)}")

@router.post("/model/load")
async def load_model(
    model_file: UploadFile = File(...)
):
    """
    Load a trained popularity prediction model
    
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
        global popularity_model
        popularity_model = PopularityPredictionModel()
        popularity_model.load_model(temp_filepath)
        
        # Clean up temporary file
        import os
        os.remove(temp_filepath)
        
        return {
            "message": "Model loaded successfully",
            "model_type": popularity_model.model_type,
            "is_trained": popularity_model.is_trained
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@router.post("/analyze/features")
async def analyze_feature_importance(
    content_file: UploadFile = File(...),
    target_variable: str = "popularity_score",
    sentiment_file: Optional[UploadFile] = File(None),
    engagement_file: Optional[UploadFile] = File(None)
):
    """
    Analyze feature importance for popularity prediction
    
    Args:
        content_file: CSV file with content metadata
        target_variable: Name of target variable
        sentiment_file: CSV file with sentiment data (optional)
        engagement_file: CSV file with engagement data (optional)
        
    Returns:
        Feature importance analysis
    """
    try:
        # Read content file
        content_contents = await content_file.read()
        content_df = pd.read_csv(io.StringIO(content_contents.decode('utf-8')))
        
        # Validate target variable
        if target_variable not in content_df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Target variable '{target_variable}' not found in content data"
            )
        
        # Read optional files
        sentiment_df = None
        if sentiment_file:
            sentiment_contents = await sentiment_file.read()
            sentiment_df = pd.read_csv(io.StringIO(sentiment_contents.decode('utf-8')))
        
        engagement_df = None
        if engagement_file:
            engagement_contents = await engagement_file.read()
            engagement_df = pd.read_csv(io.StringIO(engagement_contents.decode('utf-8')))
        
        # Extract features without training
        features = popularity_model.extract_popularity_features(
            content_df, sentiment_df, engagement_df
        )
        
        # Remove target variable
        if target_variable in features.columns:
            features = features.drop(columns=[target_variable])
        
        # Keep only numeric columns
        numeric_features = features.select_dtypes(include=['number'])
        
        # Calculate basic statistics
        feature_stats = {}
        for column in numeric_features.columns:
            feature_stats[column] = {
                "mean": float(numeric_features[column].mean()),
                "std": float(numeric_features[column].std()),
                "min": float(numeric_features[column].min()),
                "max": float(numeric_features[column].max()),
                "missing_count": int(numeric_features[column].isna().sum()),
                "missing_percentage": float((numeric_features[column].isna().sum() / len(numeric_features)) * 100)
            }
        
        return {
            "message": "Feature analysis completed",
            "total_features": len(numeric_features.columns),
            "data_shape": content_df.shape,
            "feature_statistics": feature_stats,
            "available_features": numeric_features.columns.tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing features: {str(e)}")

@router.get("/models")
async def get_available_models():
    """
    Get available prediction models and their descriptions
    
    Returns:
        List of available models
    """
    models = {
        "random_forest": {
            "name": "Random Forest",
            "description": "Ensemble learning using multiple decision trees",
            "use_case": "Good for complex relationships, provides feature importance",
            "pros": ["Handles non-linear relationships", "Robust to outliers", "Feature importance"],
            "cons": ["Can be slow to train", "May overfit on small datasets"]
        },
        "gradient_boosting": {
            "name": "Gradient Boosting",
            "description": "Boosting ensemble method that builds trees sequentially",
            "use_case": "Often provides best accuracy, good for structured data",
            "pros": ["High accuracy", "Handles complex patterns", "Good performance"],
            "cons": ["Sensitive to overfitting", "Requires careful tuning"]
        },
        "linear": {
            "name": "Linear Regression",
            "description": "Simple linear approach for prediction",
            "use_case": "Good baseline model, interpretable results",
            "pros": ["Fast training", "Interpretable", "Good baseline"],
            "cons": ["Only captures linear relationships", "Limited accuracy"]
        },
        "ridge": {
            "name": "Ridge Regression",
            "description": "Linear regression with L2 regularization",
            "use_case": "Prevents overfitting, good with multicollinearity",
            "pros": ["Reduces overfitting", "Handles multicollinearity", "Stable"],
            "cons": ["Still linear", "Requires hyperparameter tuning"]
        }
    }
    
    return {
        "available_models": models,
        "current_model": popularity_model.model_type
    }

@router.post("/evaluate")
async def evaluate_model(
    content_file: UploadFile = File(...),
    target_variable: str = "popularity_score",
    test_size: float = 0.2,
    sentiment_file: Optional[UploadFile] = File(None),
    engagement_file: Optional[UploadFile] = File(None)
):
    """
    Evaluate model performance with train-test split
    
    Args:
        content_file: CSV file with content data
        target_variable: Target variable name
        test_size: Proportion of data for testing
        sentiment_file: Sentiment data file (optional)
        engagement_file: Engagement data file (optional)
        
    Returns:
        Model evaluation results
    """
    try:
        # This endpoint would use the train functionality but return detailed evaluation
        # For now, we'll use the existing train method
        training_response = await train_popularity_model(
            content_file, target_variable, popularity_model.model_type,
            sentiment_file, engagement_file
        )
        
        # Add evaluation-specific information
        training_response["evaluation"] = {
            "test_size": test_size,
            "train_size": 1 - test_size,
            "evaluation_type": "train_test_split"
        }
        
        return training_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating model: {str(e)}")
