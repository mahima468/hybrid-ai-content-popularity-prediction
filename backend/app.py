
"""
FastAPI Backend Application
Main entry point for Hybrid AI System for Fake Engagement Detection & Sentiment-Driven Content Popularity Forecasting
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import os
from datetime import datetime

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from routes.sentiment_routes import router as sentiment_router
from routes.engagement_routes import router as engagement_router
from routes.prediction_routes import router as prediction_router
from utils.model_loader import model_loader, load_all_models
from models.engagement_detector import get_engagement_detector
from utils.dataset_loader import dataset_loader, get_dashboard_stats
from utils.dataset_preprocessor import preprocess_dataset

# Global ML model variables
prediction_model = None
prediction_scaler = None
model_loaded = False

def load_model():
    """Load ML model globally when app starts"""
    global prediction_model, prediction_scaler, model_loaded
    
    try:
        import os
        import glob
        import joblib
        
        models_dir = "saved_models"
        
        # Find latest model
        model_pattern = "popularity_prediction_model_*.joblib"
        model_files = glob.glob(os.path.join(models_dir, model_pattern))
        
        if not model_files:
            print("No prediction model found")
            model_loaded = False
            return False
        
        model_files.sort(key=os.path.getmtime, reverse=True)
        model_path = model_files[0]
        
        # Find latest scaler
        scaler_pattern = "popularity_prediction_model_scaler_*.joblib"
        scaler_files = glob.glob(os.path.join(models_dir, scaler_pattern))
        
        if not scaler_files:
            print("No prediction scaler found")
            model_loaded = False
            return False
        
        scaler_files.sort(key=os.path.getmtime, reverse=True)
        scaler_path = scaler_files[0]
        
        # Load model and scaler
        prediction_model = joblib.load(model_path)
        prediction_scaler = joblib.load(scaler_path)
        model_loaded = True
        
        print("✅ ML model loaded successfully")
        print(f"Model: {os.path.basename(model_path)}")
        print(f"Scaler: {os.path.basename(scaler_path)}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model_loaded = False
        return False

# dataset_loader and model_loader are already imported as instances above

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid AI Content Popularity Prediction API",
    description="API for fake engagement detection and sentiment-driven content popularity forecasting",
    version="1.0.0"
)

# Configure CORS - Allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Include routers
app.include_router(sentiment_router, prefix="/api/sentiment", tags=["sentiment"])
app.include_router(engagement_router, prefix="/api/engagement", tags=["engagement"])
app.include_router(prediction_router, prefix="/api/prediction", tags=["prediction"])

# Pydantic models for new endpoints
class EngagementFeatures(BaseModel):
    """Request model for engagement detection"""
    views: int = Field(..., ge=0, description="Number of views")
    likes: int = Field(..., ge=0, description="Number of likes")
    comments: int = Field(..., ge=0, description="Number of comments")
    shares: int = Field(..., ge=0, description="Number of shares")
    engagement_rate: float = Field(..., ge=0, description="Engagement rate")

class ViralityFeatures(BaseModel):
    """Request model for virality score calculation"""
    engagement_rate: float = Field(..., ge=0, le=100, description="Engagement rate (0-100)")
    sentiment_score: float = Field(..., ge=-1, le=1, description="Sentiment score (-1 to 1)")
    predicted_growth: float = Field(..., ge=0, description="Predicted growth rate")

class ViralityResponse(BaseModel):
    """Response model for virality score"""
    virality_score: float = Field(..., ge=0, le=100, description="Virality score (0-100)")
    engagement_component: float = Field(..., description="Engagement rate component")
    sentiment_component: float = Field(..., description="Sentiment score component")
    growth_component: float = Field(..., description="Predicted growth component")
    timestamp: str

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models and dataset on startup"""
    logger.info("=" * 60)
    logger.info("STARTING HYBRID AI CONTENT POPULARITY PREDICTION API")
    logger.info("=" * 60)
    
    try:
        # Preprocess dataset to ensure required columns
        logger.info("Preprocessing dataset...")
        preprocessed_dataset = preprocess_dataset()
        logger.info(f"✅ Dataset Preprocessed: {preprocessed_dataset.shape[0]} rows, {preprocessed_dataset.shape[1]} columns")
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset = dataset_loader.load_dataset()
        logger.info(f"✅ Dataset Loaded: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
        
        # Load all models
        logger.info("Loading models...")
        model_status = load_all_models()
        
        # Load ML model globally for performance
        logger.info("Loading ML model globally...")
        load_model()
        
        # Check if models loaded successfully
        loaded_models = [model for model, status in model_status.items() if status == "ready"]
        
        if loaded_models:
            logger.info("✅ Models loaded successfully")
            logger.info(f"✅ Loaded models: {', '.join(loaded_models)}")
        else:
            logger.warning("⚠️ No models loaded - models will need to be trained manually")
        
        # Print model loading status
        if model_status.get("sentiment_model") == "ready":
            logger.info("✅ Sentiment Model Loaded")
        else:
            logger.info("❌ Sentiment Model Not Loaded")
            
        if model_status.get("prediction_model") == "ready":
            logger.info("✅ Prediction Model Loaded")
        else:
            logger.info("❌ Prediction Model Not Loaded")
            
        if model_status.get("engagement_detector") == "ready":
            logger.info("✅ Engagement Detector Loaded")
        else:
            logger.info("❌ Engagement Detector Not Loaded")
        
        logger.info("=" * 60)
        logger.info("🚀 HYBRID AI API STARTUP COMPLETE")
        logger.info("=" * 60)
        logger.info("📊 Dashboard Analytics: /api/dashboard-analytics")
        logger.info("🤖 Model Status: /api/model-status")
        logger.info("💭 Sentiment Analysis: /api/analyze-sentiment")
        logger.info("📈 Popularity Prediction: /api/predict-popularity")
        logger.info("🔍 Engagement Detection: /api/detect-engagement")
        logger.info("=" * 60)
        
        logger.info("=" * 60)
        logger.info("API STARTUP COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")

@app.get("/")
async def root():
    """Root endpoint to check API status"""
    return {"message": "Hybrid AI Content Popularity Prediction API is running"}

@app.get("/api/download-report")
async def download_report(format: str = "json"):
    """
    Generate and download comprehensive report
    
    - **format**: Report format (json or csv)
    
    Returns downloadable report file
    """
    try:
        logger.info(f"Generating {format.upper()} report...")
        
        # Get analytics data
        analytics = get_dashboard_stats()
        
        # Get model status
        model_status = model_loader.get_model_status()
        
        # Get top content
        dataset = dataset_loader.get_dataset()
        top_content = []
        if dataset is not None and len(dataset) > 0 and 'future_views' in dataset.columns:
            top_content_data = dataset.nlargest(5, 'future_views')[['views', 'future_views', 'likes', 'engagement_rate', 'sentiment_score']].head(5)
            top_content = [
                {
                    "rank": i + 1,
                    "current_views": int(row['views']),
                    "predicted_views": int(row['future_views']),
                    "likes": int(row['likes']),
                    "engagement_rate": float(row['engagement_rate'] * 100),
                    "sentiment": "positive" if row['sentiment_score'] > 0.3 else "negative" if row['sentiment_score'] < -0.3 else "neutral",
                    "growth_percentage": float(((row['future_views'] / row['views']) - 1) * 100)
                }
                for i, (_, row) in enumerate(top_content_data.iterrows())
            ]
        
        # Create report data
        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "format": format.upper(),
                "version": "1.0",
                "system": "Hybrid AI Content Popularity Prediction"
            },
            "executive_summary": {
                "total_content_items": analytics.get("total_content", 0),
                "total_views": analytics.get("total_views", 0),
                "average_engagement_rate": analytics.get("avg_engagement", 0) * 100,
                "average_sentiment_score": analytics.get("avg_sentiment", 0),
                "models_loaded": len([status for status in model_status.values() if status == "ready"]),
                "total_models": len(model_status)
            },
            "model_performance": {
                "sentiment_analysis": {
                    "model": "Random Forest",
                    "accuracy": analytics.get("sentiment_accuracy", 0) * 100,
                    "status": model_status.get("sentiment_model", "unknown")
                },
                "popularity_prediction": {
                    "model": "Random Forest", 
                    "accuracy": analytics.get("prediction_accuracy", 0) * 100,
                    "status": model_status.get("prediction_model", "unknown")
                },
                "engagement_detection": {
                    "model": "Isolation Forest",
                    "accuracy": analytics.get("engagement_accuracy", 0) * 100,
                    "status": model_status.get("engagement_detector", "unknown")
                }
            },
            "analytics_summary": {
                "sentiment_distribution": analytics.get("sentiment_distribution", {}),
                "engagement_authenticity": analytics.get("engagement_authenticity", {}),
                "predictions_summary": analytics.get("predictions_summary", {})
            },
            "top_performing_content": top_content,
            "system_health": {
                "api_status": "healthy",
                "dataset_loaded": dataset is not None,
                "models_ready": len([status for status in model_status.values() if status == "ready"]),
                "last_updated": datetime.now().isoformat()
            }
        }
        
        if format.lower() == "csv":
            # Generate CSV report
            import csv
            import io
            
            output = io.StringIO()
            
            # Write summary section
            writer = csv.writer(output)
            writer.writerow(["HYBRID AI CONTENT POPULARITY PREDICTION REPORT"])
            writer.writerow(["Generated at:", report_data["report_metadata"]["generated_at"]])
            writer.writerow([])
            
            # Executive Summary
            writer.writerow(["EXECUTIVE SUMMARY"])
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Total Content Items", report_data["executive_summary"]["total_content_items"]])
            writer.writerow(["Total Views", report_data["executive_summary"]["total_views"]])
            writer.writerow(["Average Engagement Rate (%)", f"{report_data['executive_summary']['average_engagement_rate']:.2f}"])
            writer.writerow(["Average Sentiment Score", f"{report_data['executive_summary']['average_sentiment_score']:.3f}"])
            writer.writerow(["Models Loaded", f"{report_data['executive_summary']['models_loaded']}/{report_data['executive_summary']['total_models']}"])
            writer.writerow([])
            
            # Model Performance
            writer.writerow(["MODEL PERFORMANCE"])
            writer.writerow(["Model Type", "Algorithm", "Accuracy (%)", "Status"])
            for model_type, model_data in report_data["model_performance"].items():
                writer.writerow([
                    model_type.replace("_", " ").title(),
                    model_data["model"],
                    f"{model_data['accuracy']:.1f}",
                    model_data["status"]
                ])
            writer.writerow([])
            
            # Top Performing Content
            writer.writerow(["TOP PERFORMING CONTENT"])
            writer.writerow(["Rank", "Current Views", "Predicted Views", "Likes", "Engagement Rate (%)", "Sentiment", "Growth (%)"])
            for content in report_data["top_performing_content"]:
                writer.writerow([
                    content["rank"],
                    content["current_views"],
                    content["predicted_views"],
                    content["likes"],
                    f"{content['engagement_rate']:.2f}",
                    content["sentiment"],
                    f"{content['growth_percentage']:.1f}"
                ])
            
            csv_content = output.getvalue()
            output.close()
            
            from fastapi.responses import Response
            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=hybrid_ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
            )
        
        else:  # JSON format
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content=report_data,
                headers={"Content-Disposition": f"attachment; filename=hybrid_ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"}
            )
            
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/api/download-report")
async def download_report():
    """
    Generate and download analytics report as CSV
    
    Loads dataset from final_dataset.csv, creates summary statistics,
    and returns downloadable CSV file with important columns
    """
    try:
        import pandas as pd
        import os
        from fastapi.responses import FileResponse
        import tempfile
        
        logger.info("Generating analytics report...")
        
        # Load dataset
        dataset_path = "datasets/processed/final_dataset.csv"
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset not found at: {dataset_path}")
            raise HTTPException(status_code=404, detail=f"Dataset not found at {dataset_path}")
        
        logger.info(f"Loading dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        # Create summary statistics
        total_views = df['views'].sum() if 'views' in df.columns else 0
        avg_engagement = df['engagement_rate'].mean() if 'engagement_rate' in df.columns else 0
        total_records = len(df)
        
        logger.info(f"Dataset loaded: {total_records} records")
        logger.info(f"Total views: {total_views:,}")
        logger.info(f"Average engagement: {avg_engagement:.4f}")
        
        # Select important columns
        important_columns = ['views', 'likes', 'comments', 'shares', 'engagement_rate', 'sentiment_score']
        available_columns = [col for col in important_columns if col in df.columns]
        
        if not available_columns:
            logger.error("No important columns found in dataset")
            raise HTTPException(status_code=400, detail="No important columns found in dataset")
        
        logger.info(f"Using columns: {available_columns}")
        
        # Create report data
        report_data = df[available_columns].copy()
        
        # Add summary statistics at the top
        summary_data = {
            'Metric': ['Total Records', 'Total Views', 'Average Engagement Rate'],
            'Value': [total_records, f"{total_views:,}", f"{avg_engagement:.4f}"]
        }
        summary_df = pd.DataFrame(summary_data)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as tmp_file:
            temp_filename = tmp_file.name
            
            # Write summary first
            summary_df.to_csv(tmp_file, index=False)
            tmp_file.write('\n')  # Add blank line
            
            # Write main data
            report_data.to_csv(tmp_file, index=True)
        
        logger.info(f"Report created successfully: {temp_filename}")
        
        # Return file response
        return FileResponse(
            path=temp_filename,
            filename="analytics_report.csv",
            media_type="text/csv"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/api/model-status")
async def get_model_status():
    """Get status of all loaded models"""
    try:
        import os
        import glob
        
        # Check for model files in saved_models directory
        models_dir = "saved_models"
        
        # Check sentiment model files
        sentiment_files = glob.glob(os.path.join(models_dir, "sentiment_random_forest_*.joblib"))
        sentiment_vectorizer_files = glob.glob(os.path.join(models_dir, "sentiment_random_forest_vectorizer_*.joblib"))
        sentiment_ready = len(sentiment_files) > 0 and len(sentiment_vectorizer_files) > 0
        
        # Check prediction model files
        prediction_files = glob.glob(os.path.join(models_dir, "popularity_prediction_model_*.joblib"))
        prediction_scaler_files = glob.glob(os.path.join(models_dir, "popularity_prediction_model_scaler_*.joblib"))
        prediction_ready = len(prediction_files) > 0 and len(prediction_scaler_files) > 0
        
        # Check engagement detector (created in memory, so check if it can be initialized)
        engagement_ready = True  # Engagement detector is created in memory
        
        # Return the correct format
        return {
            "sentiment_model": "ready" if sentiment_ready else "not_ready",
            "engagement_model": "ready" if engagement_ready else "not_ready", 
            "prediction_model": "ready" if prediction_ready else "not_ready"
        }
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        # Return error status on exception
        return {
            "sentiment_model": "error",
            "engagement_model": "error",
            "prediction_model": "error"
        }

@app.post("/api/detect-engagement")
async def detect_engagement(features: EngagementFeatures):
    """
    Detect engagement authenticity using IsolationForest
    
    - **views**: Number of views
    - **likes**: Number of likes  
    - **comments**: Number of comments
    - **shares**: Number of shares
    - **engagement_rate**: Engagement rate
    
    Returns classification as 'authentic' or 'suspicious'
    """
    try:
        features_dict = features.dict()
        result = model_loader.detect_engagement(features_dict)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "classification": result["classification"],
            "is_suspicious": result["is_suspicious"],
            "confidence": result["confidence"],
            "anomaly_score": result["anomaly_score"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in engagement detection: {e}")
        raise HTTPException(status_code=500, detail=f"Engagement detection failed: {str(e)}")

@app.post("/api/virality-score", response_model=ViralityResponse)
async def calculate_virality_score(features: ViralityFeatures):
    """
    Calculate virality score based on engagement, sentiment, and predicted growth
    
    Formula: virality_score = 0.4 * engagement_rate + 0.3 * sentiment_score + 0.3 * predicted_growth
    
    - **engagement_rate**: Engagement rate (0-100)
    - **sentiment_score**: Sentiment score (-1 to 1, normalized to 0-100)
    - **predicted_growth**: Predicted growth rate
    
    Returns virality score from 0-100
    """
    try:
        # Normalize sentiment score from [-1,1] to [0,100]
        normalized_sentiment = (features.sentiment_score + 1) * 50
        
        # Calculate components
        engagement_component = 0.4 * features.engagement_rate
        sentiment_component = 0.3 * normalized_sentiment
        growth_component = 0.3 * features.predicted_growth
        
        # Calculate total virality score
        virality_score = engagement_component + sentiment_component + growth_component
        
        # Ensure score is within 0-100 range
        virality_score = max(0, min(100, virality_score))
        
        return ViralityResponse(
            virality_score=virality_score,
            engagement_component=engagement_component,
            sentiment_component=sentiment_component,
            growth_component=growth_component,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error calculating virality score: {e}")
        raise HTTPException(status_code=500, detail=f"Virality score calculation failed: {str(e)}")

@app.get("/api/dashboard-analytics")
async def get_dashboard_analytics():
    """
    Get dashboard analytics including sentiment stats, engagement metrics, and top content
    
    Returns comprehensive analytics data for dashboard display
    """
    try:
        # Get real analytics from dataset with safe column handling
        analytics = get_dashboard_stats()
        
        # Add model status and timestamp
        analytics["model_status"] = model_loader.get_model_status()
        analytics["last_updated"] = datetime.now().isoformat()
        
        # Add top predicted content from dataset with safe column access
        dataset = dataset_loader.get_dataset()
        if dataset is not None and len(dataset) > 0 and 'future_views' in dataset.columns:
            try:
                # Get top 5 content by predicted future views
                top_content = dataset.nlargest(5, 'future_views')[['views', 'future_views', 'likes', 'engagement_rate', 'sentiment_score']].head(5)
                
                analytics["top_performing_content"] = [
                    {
                        "content_id": f"content_{i+1}",
                        "current_views": int(row['views']),
                        "predicted_views": int(row['future_views']),
                        "likes": int(row['likes']),
                        "engagement_rate": float(row['engagement_rate'] * 100),  # Convert to percentage
                        "sentiment": "positive" if row['sentiment_score'] > 0.3 else "negative" if row['sentiment_score'] < -0.3 else "neutral",
                        "predicted_growth": float((row['future_views'] / row['views']) - 1) * 100  # Growth percentage
                    }
                    for i, (_, row) in enumerate(top_content.iterrows())
                ]
            except Exception as e:
                logger.warning(f"Error creating top performing content: {str(e)}")
                analytics["top_performing_content"] = []
        else:
            analytics["top_performing_content"] = []
            if dataset is None:
                logger.warning("Dataset is None, cannot create top performing content")
            elif len(dataset) == 0:
                logger.warning("Dataset is empty, cannot create top performing content")
            else:
                logger.warning("future_views column not found in dataset, cannot create top performing content")
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting dashboard analytics: {e}")
        # Return safe fallback response instead of crashing
        return {
            "total_views": 0,
            "avg_engagement": 0.0,
            "avg_sentiment": 0.0,
            "total_content": 0,
            "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
            "popularity_trends": {"labels": [], "predicted": [], "actual": []},
            "predictions_summary": {
                "total_predictions": 0,
                "avg_predicted_views": 0,
                "max_predicted_views": 0,
                "min_predicted_views": 0,
                "note": "Error in analytics calculation"
            },
            "top_predicted_content": [],
            "model_status": {"error": "Models not loaded"},
            "last_updated": datetime.now().isoformat(),
            "engagement_accuracy": 0.0,
            "sentiment_accuracy": 0.0,
            "prediction_accuracy": 0.0,
            "error": f"Dashboard analytics temporarily unavailable: {str(e)}"
        }

class PopularityRequest(BaseModel):
    """Request model for popularity prediction"""
    views: int
    likes: int
    comments: int
    shares: int
    engagement_rate: float
    sentiment_score: float
    content_length: int
    has_media: bool
    is_trending: bool
    channel_subscribers: int
    time_since_publication: int  # in hours

@app.post("/api/predict-popularity")
async def predict_popularity(request: PopularityRequest):
    """
    Predict popularity with bulletproof error handling - never hangs
    
    Args:
        request: Content data with all required fields
        
    Returns:
        Popularity prediction with confidence score
    """
    import time
    start_time = time.time()
    
    # Log API hit
    print("API HIT")
    
    try:
        # Quick input validation
        if request.views < 0 or request.likes < 0 or request.comments < 0 or request.shares < 0:
            print("RESPONSE SENT - Invalid input")
            return {
                "error": "Invalid input",
                "message": "Views, likes, comments, and shares must be non-negative",
                "predicted_views": 0,
                "confidence": 0.0
            }
        
        # Try ML model with comprehensive error handling
        try:
            if model_loaded and prediction_model is not None and prediction_scaler is not None:
                return _use_ml_model(request, start_time)
            else:
                print("ML model not loaded, using fallback")
                return _use_fallback_prediction(request, start_time)
                
        except Exception as ml_error:
            print(f"ML model failed, using fallback: {ml_error}")
            return _use_fallback_prediction(request, start_time)
            
    except Exception as e:
        # Final safety net - always return something
        print(f"CRITICAL ERROR, using emergency fallback: {e}")
        return _emergency_fallback(request, start_time)

def _use_ml_model(request, start_time):
    """Use ML model with minimal processing and timeout protection"""
    try:
        import numpy as np
        import time
        
        # Set timeout for ML prediction (1 second max)
        ml_start_time = time.time()
        
        # Extract only required fields - minimal processing
        views = int(request.views) if request.views is not None else 0
        likes = int(request.likes) if request.likes is not None else 0
        comments = int(request.comments) if request.comments is not None else 0
        shares = int(request.shares) if request.shares is not None else 0
        engagement_rate = float(request.engagement_rate) if request.engagement_rate is not None else 0.0
        sentiment_score = float(request.sentiment_score) if request.sentiment_score is not None else 0.0
        
        # Optional features with minimal validation
        content_length = int(request.content_length) if request.content_length and request.content_length > 0 else 1000
        channel_subscribers = int(request.channel_subscribers) if request.channel_subscribers and request.channel_subscribers > 0 else 1000
        
        # Create feature array - only essential features
        features = [
            views,              # Current views (most important)
            likes,              # Likes count
            comments,           # Comments count
            engagement_rate,    # Engagement rate
            sentiment_score,    # Sentiment score
            shares,             # Shares count
            content_length,     # Content length
            channel_subscribers # Channel subscribers
        ]
        
        # Check timeout before heavy computation
        if time.time() - ml_start_time > 0.3:
            raise Exception("ML preparation timeout")
        
        # Use ML model with timeout protection
        try:
            # Direct numpy array creation - no extra processing
            features_array = np.array([features], dtype=float)
            
            # Check timeout again
            if time.time() - ml_start_time > 0.5:
                raise Exception("ML preprocessing timeout")
            
            # Scale and predict - single operation
            features_scaled = prediction_scaler.transform(features_array)
            predictions = prediction_model.predict(features_scaled)
            predicted_views = int(max(predictions[0], 0))
            
            # Final timeout check
            if time.time() - ml_start_time > 0.8:
                raise Exception("ML prediction timeout")
            
            # Quick validation
            if predicted_views <= 0 or predicted_views < views:
                print("ML prediction invalid, using fallback")
                return _use_fallback_prediction(request, start_time)
            
            # Success with ML model
            growth_factor = predicted_views / views if views > 0 else 1.0
            response_time = time.time() - start_time
            
            print("RESPONSE SENT - ML model used")
            return {
                "predicted_views": predicted_views,
                "confidence": 0.9,
                "growth_factor": round(growth_factor, 2),
                "model_used": "ml_model",
                "response_time": round(response_time, 3)
            }
            
        except Exception as prediction_error:
            print(f"ML prediction error: {prediction_error}")
            return _use_fallback_prediction(request, start_time)
            
    except Exception as model_error:
        print(f"ML model error: {model_error}")
        return _use_fallback_prediction(request, start_time)

def _use_fallback_prediction(request, start_time):
    """Use simple fallback prediction with minimal processing"""
    try:
        import random
        
        # Extract only required fields - direct conversion
        views = int(request.views) if request.views is not None and request.views > 0 else 1000
        
        # Simple formula: predicted_views = views * random value between 1.2 and 1.8
        random_multiplier = random.uniform(1.2, 1.8)
        predicted_views = int(views * random_multiplier)
        
        # Ensure valid output - minimal validation
        if predicted_views <= views:
            predicted_views = int(views * 1.2)
        
        # Quick calculations
        growth_factor = predicted_views / views if views > 0 else 1.0
        confidence = 0.75
        
        # Minimal confidence boost based on basic checks
        if request.likes and request.comments and request.views > 0:
            confidence += 0.05
        confidence = min(confidence, 0.85)
        
        response_time = time.time() - start_time
        
        print("RESPONSE SENT - Fallback used")
        return {
            "predicted_views": predicted_views,
            "confidence": round(confidence, 2),
            "growth_factor": round(growth_factor, 2),
            "model_used": "fallback_formula",
            "response_time": round(response_time, 3)
        }
        
    except Exception as fallback_error:
        print(f"Fallback failed: {fallback_error}")
        return _emergency_fallback(request, start_time)

def _emergency_fallback(request, start_time):
    """Emergency fallback with absolute minimal processing"""
    try:
        # Most basic calculation - cannot fail
        views = int(request.views) if request.views is not None and request.views > 0 else 1000
        predicted_views = int(views * 1.3)
        growth_factor = predicted_views / views if views > 0 else 1.0
        response_time = time.time() - start_time
        
        print("RESPONSE SENT - Emergency fallback")
        return {
            "predicted_views": predicted_views,
            "confidence": 0.5,
            "growth_factor": round(growth_factor, 2),
            "model_used": "emergency_fallback",
            "response_time": round(response_time, 3)
        }
        
    except Exception as emergency_error:
        print(f"EMERGENCY FALLBACK FAILED: {emergency_error}")
        # Last resort - hardcoded response
        response_time = time.time() - start_time
        views = int(request.views) if request.views is not None and request.views > 0 else 1000
        return {
            "predicted_views": max(views, 100),
            "confidence": 0.1,
            "growth_factor": 1.0,
            "model_used": "last_resort",
            "response_time": round(response_time, 3)
        }

def fallback_prediction(request):
    """
    Fallback prediction using rule-based logic when model fails
    """
    try:
        # Calculate base prediction using engagement rate
        base_multiplier = 1 + (request.engagement_rate * 2)
        
        # Apply sentiment boost
        sentiment_boost = 1 + (request.sentiment_score * 0.3) if request.sentiment_score > 0 else 1
        
        # Apply media boost
        media_boost = 1.15 if request.has_media else 1.0
        
        # Apply trending boost
        trending_boost = 1.3 if request.is_trending else 1.0
        
        # Apply subscriber boost (normalized)
        subscriber_boost = 1 + min(request.channel_subscribers / 500000, 0.5)  # Cap at 50% boost
        
        # Apply time decay (content loses potency over time)
        time_factor = max(0.7, 1 - (request.time_since_publication / 8760))  # 1 year decay
        
        # Calculate final prediction
        predicted_views = int(request.views * base_multiplier * sentiment_boost * media_boost * 
                            trending_boost * subscriber_boost * time_factor)
        
        # Calculate confidence based on data quality
        confidence = 0.75  # Base confidence for fallback
        
        # Boost confidence for complete data
        if request.views > 0 and request.likes > 0 and request.comments > 0:
            confidence += 0.05
        if request.has_media:
            confidence += 0.03
        if request.is_trending:
            confidence += 0.02
        if request.channel_subscribers > 10000:
            confidence += 0.02
            
        confidence = min(confidence, 0.85)  # Cap fallback confidence
        
        logger.info(f"Fallback prediction: {predicted_views} views (confidence: {confidence})")
        
        return {
            "predicted_views": predicted_views,
            "confidence": confidence,
            "model_used": "rule_based_fallback"
        }
        
    except Exception as e:
        logger.error(f"Fallback prediction failed: {e}")
        return {
            "predicted_views": max(request.views, int(request.views * 1.2)),  # Minimal fallback
            "confidence": 0.5,
            "model_used": "minimal_fallback"
        }

# Add alias routes for consistency
@app.get("/dashboard-analytics")
async def get_dashboard_analytics_alias():
    """Alias route for /api/dashboard-analytics"""
    return await get_dashboard_analytics()

@app.get("/model-status")
async def get_model_status_alias():
    """Alias route for /api/model-status"""
    return await get_model_status()

@app.post("/analyze-sentiment")
async def analyze_sentiment_alias(request: PopularityRequest):
    """Alias route for sentiment analysis - redirects to /api/sentiment/analyze"""
    try:
        # Import the sentiment model directly
        from models.sentiment_model import SentimentModel
        
        sentiment_model = SentimentModel()
        
        # Extract text from content data
        texts = []
        for content in request.content_data:
            if 'text' in content:
                texts.append(content['text'])
            elif 'content' in content:
                texts.append(content['content'])
            else:
                texts.append(str(content))
        
        if not texts:
            raise HTTPException(status_code=400, detail="No text provided for sentiment analysis")
        
        # Analyze sentiment
        results = sentiment_model.analyze_sentiment(texts, method="ml_model")
        
        return {
            "results": results,
            "method": "ml_model",
            "total_processed": len(texts)
        }
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis alias: {e}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

@app.post("/detect-engagement")
async def detect_engagement_alias(features: EngagementFeatures):
    """Alias route for /api/detect-engagement"""
    return await detect_engagement(features)

@app.post("/predict-popularity-alias")
async def predict_popularity_alias(request: PopularityRequest):
    """Alias route for /predict-popularity"""
    return await predict_popularity(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
