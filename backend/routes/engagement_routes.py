"""
Fake Engagement Detection API Routes
Provides endpoints for fake engagement detection functionality
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import json
from models.engagement_detection import EngagementDetectionModel
import io

router = APIRouter()

# Global model instance
engagement_model = EngagementDetectionModel()

class EngagementDetectionRequest(BaseModel):
    detection_method: str = "isolation_forest"  # "isolation_forest", "random_forest", "dbscan"

class EngagementData(BaseModel):
    engagement_data: List[Dict[str, Any]]

@router.post("/detect")
async def detect_fake_engagement(
    request: EngagementDetectionRequest,
    data: EngagementData
):
    """
    Detect fake engagement patterns in engagement data
    
    Args:
        request: Detection method configuration
        data: Engagement data to analyze
        
    Returns:
        Detection results with anomaly scores
    """
    try:
        if not data.engagement_data:
            raise HTTPException(status_code=400, detail="No engagement data provided")
        
        # Create new model instance with specified method
        global engagement_model
        engagement_model = EngagementDetectionModel(detection_method=request.detection_method)
        
        # Convert to DataFrame
        df = pd.DataFrame(data.engagement_data)
        
        # Detect anomalies
        detection_results = engagement_model.detect_anomalies(df)
        
        return {
            "message": "Fake engagement detection completed",
            "detection_method": request.detection_method,
            "results": detection_results["results"].to_dict('records'),
            "summary": detection_results["summary"],
            "feature_importance": detection_results["feature_importance"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting fake engagement: {str(e)}")

@router.post("/detect/file")
async def detect_fake_engagement_from_file(
    data_file: UploadFile = File(...),
    detection_method: str = "isolation_forest"
):
    """
    Detect fake engagement patterns from uploaded CSV file
    
    Args:
        data_file: CSV file containing engagement data
        detection_method: Method to use for detection
        
    Returns:
        Detection results
    """
    try:
        # Read uploaded file
        contents = await data_file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Create new model instance
        global engagement_model
        engagement_model = EngagementDetectionModel(detection_method=detection_method)
        
        # Detect anomalies
        detection_results = engagement_model.detect_anomalies(df)
        
        return {
            "message": "Fake engagement detection completed",
            "detection_method": detection_method,
            "data_shape": df.shape,
            "results": detection_results["results"].to_dict('records'),
            "summary": detection_results["summary"],
            "feature_importance": detection_results["feature_importance"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.post("/train")
async def train_engagement_model(
    training_data: UploadFile = File(...),
    detection_method: str = "random_forest",
    label_column: str = "is_fake"
):
    """
    Train the engagement detection model with labeled data
    
    Args:
        training_data: CSV file with labeled training data
        detection_method: Method to use (must be 'random_forest' for supervised training)
        label_column: Column name containing labels (0: real, 1: fake)
        
    Returns:
        Training results and metrics
    """
    try:
        if detection_method != "random_forest":
            raise HTTPException(
                status_code=400, 
                detail="Only 'random_forest' method supports supervised training"
            )
        
        # Read uploaded file
        contents = await training_data.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate label column
        if label_column not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Label column '{label_column}' not found in data"
            )
        
        # Create new model instance
        global engagement_model
        engagement_model = EngagementDetectionModel(detection_method=detection_method)
        
        # Train model
        labels = df[label_column].tolist()
        training_results = engagement_model.train_supervised(df, labels)
        
        return {
            "message": "Model trained successfully",
            "detection_method": detection_method,
            "training_samples": len(df),
            "metrics": training_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@router.get("/model/status")
async def get_model_status():
    """
    Get the current status of the engagement detection model
    
    Returns:
        Model status information
    """
    return {
        "is_trained": engagement_model.is_trained,
        "detection_method": engagement_model.detection_method,
        "model_loaded": engagement_model.model is not None
    }

@router.post("/model/save")
async def save_model(filename: str = "engagement_model.joblib"):
    """
    Save the trained engagement detection model
    
    Args:
        filename: Name for the saved model file
        
    Returns:
        Save confirmation
    """
    try:
        if engagement_model.detection_method == "dbscan":
            raise HTTPException(
                status_code=400, 
                detail="DBSCAN model cannot be saved as it doesn't have persistent state"
            )
        
        filepath = f"saved_models/{filename}"
        engagement_model.save_model(filepath)
        
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
    Load a trained engagement detection model
    
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
        global engagement_model
        engagement_model = EngagementDetectionModel()
        engagement_model.load_model(temp_filepath)
        
        # Clean up temporary file
        import os
        os.remove(temp_filepath)
        
        return {
            "message": "Model loaded successfully",
            "detection_method": engagement_model.detection_method,
            "is_trained": engagement_model.is_trained
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@router.post("/analyze/patterns")
async def analyze_engagement_patterns(
    data_file: UploadFile = File(...),
    analysis_type: str = "comprehensive"
):
    """
    Analyze engagement patterns in the data
    
    Args:
        data_file: CSV file containing engagement data
        analysis_type: Type of analysis ('comprehensive', 'temporal', 'user_behavior')
        
    Returns:
        Pattern analysis results
    """
    try:
        # Read uploaded file
        contents = await data_file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        analysis_results = {}
        
        if analysis_type in ["comprehensive", "temporal"]:
            # Temporal analysis
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                
                temporal_stats = {
                    "hourly_distribution": df['hour'].value_counts().to_dict(),
                    "daily_distribution": df['day_of_week'].value_counts().to_dict(),
                    "peak_hour": int(df['hour'].mode().iloc[0]) if not df['hour'].mode().empty else None,
                    "peak_day": int(df['day_of_week'].mode().iloc[0]) if not df['day_of_week'].mode().empty else None
                }
                analysis_results["temporal_analysis"] = temporal_stats
        
        if analysis_type in ["comprehensive", "user_behavior"]:
            # User behavior analysis
            if 'user_id' in df.columns:
                user_stats = df.groupby('user_id').agg({
                    'user_id': 'count',
                    **{col: 'sum' for col in df.columns if 'count' in col.lower()}
                })
                user_stats.columns = ['total_actions'] + [f"total_{col}" for col in user_stats.columns[1:]]
                
                user_behavior_stats = {
                    "unique_users": len(user_stats),
                    "avg_actions_per_user": float(user_stats['total_actions'].mean()),
                    "most_active_user": str(user_stats['total_actions'].idxmax()),
                    "max_actions_by_user": int(user_stats['total_actions'].max())
                }
                analysis_results["user_behavior_analysis"] = user_behavior_stats
        
        # General statistics
        general_stats = {
            "total_engagements": len(df),
            "unique_content": df['content_id'].nunique() if 'content_id' in df.columns else None,
            "unique_users": df['user_id'].nunique() if 'user_id' in df.columns else None,
            "date_range": {
                "start": df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
                "end": df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None
            }
        }
        analysis_results["general_statistics"] = general_stats
        
        return {
            "message": "Pattern analysis completed",
            "analysis_type": analysis_type,
            "data_shape": df.shape,
            "results": analysis_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing patterns: {str(e)}")

@router.get("/methods")
async def get_available_methods():
    """
    Get available detection methods and their descriptions
    
    Returns:
        List of available methods
    """
    methods = {
        "isolation_forest": {
            "name": "Isolation Forest",
            "description": "Unsupervised anomaly detection using isolation trees",
            "use_case": "Best for detecting outliers without labeled data",
            "requires_training": False
        },
        "random_forest": {
            "name": "Random Forest",
            "description": "Supervised learning using ensemble of decision trees",
            "use_case": "Best when you have labeled data (real vs fake engagement)",
            "requires_training": True
        },
        "dbscan": {
            "name": "DBSCAN",
            "description": "Density-based spatial clustering for anomaly detection",
            "use_case": "Good for detecting clusters of normal behavior and outliers",
            "requires_training": False
        }
    }
    
    return {
        "available_methods": methods,
        "current_method": engagement_model.detection_method
    }
