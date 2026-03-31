"""
Model Loader Utility
Automatically loads latest trained models from backend/saved_models
"""

import os
import glob
import joblib
import logging
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Model Loader Class
    Handles loading and management of trained models
    """
    
    def __init__(self, models_dir: str = "saved_models"):
        """
        Initialize the model loader
        
        Args:
            models_dir (str): Directory containing saved models
        """
        self.models_dir = models_dir
        self.sentiment_model = None
        self.sentiment_vectorizer = None
        self.popularity_model = None
        self.popularity_scaler = None
        self.engagement_detector = None
        
        # Model status
        self.model_status = {
            "sentiment_model": "not_ready",
            "prediction_model": "not_ready",
            "engagement_detector": "not_ready"
        }
        
        logger.info(f"Model loader initialized. Models directory: {models_dir}")
    
    def validate_datasets(self) -> Dict[str, bool]:
        """
        Validate that required datasets are available
        
        Returns:
            Dict[str, bool]: Dataset availability status
        """
        dataset_status = {}
        
        # Check sentiment training dataset
        sentiment_dataset = "../datasets/processed/sentiment_training_data.csv"
        dataset_status["sentiment_dataset"] = os.path.exists(sentiment_dataset)
        
        # Check final dataset for prediction
        final_dataset = "../datasets/processed/final_dataset.csv"
        dataset_status["final_dataset"] = os.path.exists(final_dataset)
        
        # Check raw Twitter dataset
        twitter_dataset = "../datasets/external/twitter_sentiment.csv"
        dataset_status["twitter_dataset"] = os.path.exists(twitter_dataset)
        
        # Log dataset status
        logger.info("=== Dataset Availability Check ===")
        for dataset, available in dataset_status.items():
            status = "✅ Available" if available else "❌ Missing"
            logger.info(f"{dataset}: {status}")
        
        return dataset_status
    
    def find_latest_file(self, pattern: str) -> Optional[str]:
        """
        Find the latest file matching a pattern
        
        Args:
            pattern (str): Glob pattern to match files
            
        Returns:
            Optional[str]: Path to the latest file
        """
        full_pattern = os.path.join(self.models_dir, pattern)
        files = glob.glob(full_pattern)
        
        if not files:
            return None
        
        # Sort by modification time (most recent first)
        files.sort(key=os.path.getmtime, reverse=True)
        latest_file = files[0]
        logger.info(f"Found latest file for pattern '{pattern}': {os.path.basename(latest_file)}")
        return latest_file
    
    def load_sentiment_model(self) -> bool:
        """
        Load the latest sentiment analysis model and vectorizer
        
        Returns:
            bool: True if loaded successfully
        """
        try:
            # Find latest sentiment model
            model_pattern = "sentiment_random_forest_*.joblib"
            model_path = self.find_latest_file(model_pattern)
            
            if not model_path:
                logger.warning("No sentiment model found")
                self.model_status["sentiment_model"] = "not_ready"
                return False
            
            # Find corresponding vectorizer
            vectorizer_pattern = "sentiment_random_forest_vectorizer_*.joblib"
            vectorizer_path = self.find_latest_file(vectorizer_pattern)
            
            if not vectorizer_path:
                logger.warning("No sentiment vectorizer found")
                self.model_status["sentiment_model"] = "not_ready"
                return False
            
            # Load model and vectorizer
            logger.info(f"Loading sentiment model from: {os.path.basename(model_path)}")
            self.sentiment_model = joblib.load(model_path)
            
            logger.info(f"Loading sentiment vectorizer from: {os.path.basename(vectorizer_path)}")
            self.sentiment_vectorizer = joblib.load(vectorizer_path)
            
            self.model_status["sentiment_model"] = "ready"
            logger.info("✅ Sentiment model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading sentiment model: {str(e)}")
            self.model_status["sentiment_model"] = "error"
            return False
    
    def load_prediction_model(self) -> bool:
        """
        Load the latest popularity prediction model and scaler
        
        Returns:
            bool: True if loaded successfully
        """
        try:
            # Find latest popularity model
            model_pattern = "popularity_prediction_model_*.joblib"
            model_path = self.find_latest_file(model_pattern)
            
            if not model_path:
                logger.warning("No popularity prediction model found")
                self.model_status["prediction_model"] = "not_ready"
                return False
            
            # Find corresponding scaler
            scaler_pattern = "popularity_prediction_model_scaler_*.joblib"
            scaler_path = self.find_latest_file(scaler_pattern)
            
            if not scaler_path:
                logger.warning("No popularity scaler found")
                self.model_status["prediction_model"] = "not_ready"
                return False
            
            # Load model and scaler
            logger.info(f"Loading popularity model from: {os.path.basename(model_path)}")
            self.popularity_model = joblib.load(model_path)
            
            logger.info(f"Loading popularity scaler from: {os.path.basename(scaler_path)}")
            self.popularity_scaler = joblib.load(scaler_path)
            
            self.model_status["prediction_model"] = "ready"
            logger.info("✅ Popularity prediction model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading popularity prediction model: {str(e)}")
            self.model_status["prediction_model"] = "error"
            return False
    
    def load_engagement_detector(self) -> bool:
        """
        Initialize engagement detector
        
        Returns:
            bool: True if initialized successfully
        """
        try:
            # Import here to avoid circular imports
            from models.engagement_detector import EngagementDetector
            
            logger.info("Initializing engagement detector...")
            self.engagement_detector = EngagementDetector()
            self.model_status["engagement_detector"] = "ready"
            logger.info("✅ Engagement detector initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing engagement detector: {str(e)}")
            self.model_status["engagement_detector"] = "error"
            return False
    
    def load_all_models(self) -> Dict[str, str]:
        """
        Load all available models
        
        Returns:
            Dict[str, str]: Dictionary with model status
        """
        logger.info("=" * 50)
        logger.info("LOADING ALL MODELS")
        logger.info("=" * 50)
        
        # Validate datasets first
        dataset_status = self.validate_datasets()
        
        # Check if critical datasets are missing
        if not dataset_status.get("sentiment_dataset", False):
            logger.warning("⚠️  Sentiment training dataset not found. Models may not be available.")
        if not dataset_status.get("final_dataset", False):
            logger.warning("⚠️  Final dataset not found. Prediction model may not be available.")
        
        # Load sentiment model
        self.load_sentiment_model()
        
        # Load prediction model
        self.load_prediction_model()
        
        # Initialize engagement detector
        self.load_engagement_detector()
        
        # Print summary
        logger.info("=" * 50)
        logger.info("MODEL LOADING SUMMARY")
        logger.info("=" * 50)
        
        for model_type, status in self.model_status.items():
            status_icon = "✅" if status == "ready" else "❌"
            logger.info(f"{model_type.replace('_', ' ').title()}: {status_icon} {status}")
        
        loaded_count = sum(1 for status in self.model_status.values() if status == "ready")
        total_count = len(self.model_status)
        logger.info(f"\nTotal: {loaded_count}/{total_count} models ready")
        
        return self.model_status.copy()
    
    def get_model_status(self) -> Dict[str, str]:
        """
        Get current model status
        
        Returns:
            Dict[str, str]: Dictionary with model status
        """
        return self.model_status.copy()
    
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment using loaded model
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        if self.sentiment_model is None or self.sentiment_vectorizer is None:
            return {
                "error": "Sentiment model not loaded",
                "sentiment": "unknown",
                "confidence": 0.0
            }
        
        try:
            # Preprocess text
            import re
            cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            
            # Make prediction
            prediction = self.sentiment_model.predict([cleaned_text])[0]
            probabilities = self.sentiment_model.predict_proba([cleaned_text])[0]
            
            # Map prediction to sentiment name
            sentiment_names = {0: 'negative', 1: 'neutral', 2: 'positive'}
            sentiment_name = sentiment_names.get(prediction, 'unknown')
            
            return {
                "text": text,
                "sentiment_label": prediction,
                "sentiment_name": sentiment_name,
                "confidence": max(probabilities),
                "probabilities": {
                    "negative": probabilities[0],
                    "neutral": probabilities[1],
                    "positive": probabilities[2]
                }
            }
            
        except Exception as e:
            return {
                "error": f"Prediction failed: {str(e)}",
                "sentiment": "error",
                "confidence": 0.0
            }
    
    def predict_popularity(self, features: List[List[float]]) -> List[float]:
        """
        Predict popularity using loaded RandomForest model
        
        Args:
            features (List[List[float]]): Input feature vectors
            
        Returns:
            List[float]: Predicted future views
        """
        if self.popularity_model is None or self.popularity_scaler is None:
            logger.error("Popularity model or scaler not loaded")
            return []
        
        try:
            # Convert to numpy array
            features_array = np.array(features)
            
            # Handle missing values - replace with 0
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ensure all features are numeric
            features_array = features_array.astype(float)
            
            # Scale features using the loaded scaler
            features_scaled = self.popularity_scaler.transform(features_array)
            
            # Make prediction using RandomForest
            predictions = self.popularity_model.predict(features_scaled)
            
            # Ensure predictions are positive
            predictions = np.maximum(predictions, 0)
            
            logger.info(f"Model prediction successful for {len(features)} samples")
            return predictions.tolist()
            
        except Exception as e:
            logger.error(f"Error in popularity prediction: {str(e)}")
            return []
    
    def detect_engagement(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect engagement authenticity using loaded detector
        
        Args:
            features (Dict[str, Any]): Input features
            
        Returns:
            Dict[str, Any]: Detection results
        """
        if self.engagement_detector is None:
            return {
                "error": "Engagement detector not loaded",
                "classification": "unknown"
            }
        
        try:
            return self.engagement_detector.detect(features)
        except Exception as e:
            return {
                "error": f"Detection failed: {str(e)}",
                "classification": "error"
            }

# Global model loader instance
model_loader = ModelLoader()

# Convenience functions
def load_sentiment_model() -> bool:
    """Load sentiment model using global loader"""
    return model_loader.load_sentiment_model()

def load_prediction_model() -> bool:
    """Load prediction model using global loader"""
    return model_loader.load_prediction_model()

def get_model_status() -> Dict[str, str]:
    """Get model status using global loader"""
    return model_loader.get_model_status()

def load_all_models() -> Dict[str, str]:
    """Load all models using global loader"""
    return model_loader.load_all_models()
