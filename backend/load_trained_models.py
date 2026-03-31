"""
Model Loading Utility
Loads trained models for the Hybrid AI Content Popularity Prediction System
"""

import os
import joblib
import json
import glob
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class ModelLoader:
    """
    Model Loading Class
    Handles loading and management of trained models
    """
    
    def __init__(self, models_dir: str = 'saved_models'):
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
        self.popularity_features = None
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        print(f"Model loader initialized. Models directory: {models_dir}")
    
    def find_latest_model(self, model_pattern: str) -> Optional[str]:
        """
        Find the latest model file matching a pattern
        
        Args:
            model_pattern (str): Pattern to match model files
            
        Returns:
            Optional[str]: Path to the latest model file
        """
        pattern = os.path.join(self.models_dir, model_pattern)
        model_files = glob.glob(pattern)
        
        if not model_files:
            return None
        
        # Sort by modification time (most recent first)
        model_files.sort(key=os.path.getmtime, reverse=True)
        return model_files[0]
    
    def load_sentiment_model(self) -> bool:
        """
        Load the latest sentiment analysis model
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Find latest sentiment model
            model_path = self.find_latest_model('sentiment_*_*.joblib')
            if not model_path:
                print("No sentiment model found in saved_models directory")
                return False
            
            print(f"Loading sentiment model from: {model_path}")
            self.sentiment_model = joblib.load(model_path)
            
            # Find corresponding vectorizer
            vectorizer_path = self.find_latest_model('sentiment_*_vectorizer_*.joblib')
            if vectorizer_path:
                print(f"Loading sentiment vectorizer from: {vectorizer_path}")
                self.sentiment_vectorizer = joblib.load(vectorizer_path)
            else:
                print("Warning: No sentiment vectorizer found")
                return False
            
            # Load metadata
            metadata_path = self.find_latest_model('sentiment_*_metadata_*.json')
            if metadata_path:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"Sentiment model metadata: {metadata}")
            
            print("✅ Sentiment model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading sentiment model: {str(e)}")
            return False
    
    def load_popularity_model(self) -> bool:
        """
        Load the latest popularity prediction model
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Find latest popularity model
            model_path = self.find_latest_model('popularity_prediction_model_*_*.joblib')
            if not model_path:
                print("No popularity prediction model found in saved_models directory")
                return False
            
            print(f"Loading popularity model from: {model_path}")
            self.popularity_model = joblib.load(model_path)
            
            # Find corresponding scaler
            scaler_path = self.find_latest_model('popularity_prediction_model_*_scaler_*.joblib')
            if scaler_path:
                print(f"Loading popularity scaler from: {scaler_path}")
                self.popularity_scaler = joblib.load(scaler_path)
            else:
                print("Warning: No popularity scaler found")
                return False
            
            # Find feature columns
            features_path = self.find_latest_model('popularity_prediction_model_*_features_*.json')
            if features_path:
                with open(features_path, 'r') as f:
                    self.popularity_features = json.load(f)
                print(f"Loaded {len(self.popularity_features)} features")
            else:
                print("Warning: No feature list found")
                return False
            
            # Load metadata
            metadata_path = self.find_latest_model('popularity_prediction_model_*_metadata_*.json')
            if metadata_path:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"Popularity model metadata: {metadata}")
            
            print("✅ Popularity prediction model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading popularity model: {str(e)}")
            return False
    
    def load_all_models(self) -> Dict[str, bool]:
        """
        Load all available models
        
        Returns:
            Dict[str, bool]: Dictionary indicating which models were loaded
        """
        print("=" * 50)
        print("LOADING TRAINED MODELS")
        print("=" * 50)
        
        results = {
            'sentiment': self.load_sentiment_model(),
            'popularity': self.load_popularity_model()
        }
        
        print("\n" + "=" * 50)
        print("MODEL LOADING SUMMARY")
        print("=" * 50)
        
        for model_type, loaded in results.items():
            status = "✅ LOADED" if loaded else "❌ NOT FOUND"
            print(f"{model_type.capitalize()} Model: {status}")
        
        loaded_count = sum(results.values())
        total_count = len(results)
        print(f"\nTotal: {loaded_count}/{total_count} models loaded")
        
        return results
    
    def get_sentiment_prediction(self, text: str) -> Dict[str, Any]:
        """
        Get sentiment prediction using loaded model
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        if self.sentiment_model is None or self.sentiment_vectorizer is None:
            return {
                'error': 'Sentiment model not loaded',
                'sentiment': 'unknown',
                'confidence': 0.0
            }
        
        try:
            # Preprocess text (basic cleaning)
            import re
            cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            
            # Make prediction
            prediction = self.sentiment_model.predict([cleaned_text])[0]
            probabilities = self.sentiment_model.predict_proba([cleaned_text])[0]
            
            # Map prediction to sentiment name
            sentiment_names = {0: 'negative', 1: 'neutral', 2: 'positive'}
            sentiment_name = sentiment_names.get(prediction, 'unknown')
            
            return {
                'text': text,
                'sentiment_label': prediction,
                'sentiment_name': sentiment_name,
                'confidence': max(probabilities),
                'probabilities': {
                    'negative': probabilities[0],
                    'neutral': probabilities[1],
                    'positive': probabilities[2]
                }
            }
            
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'sentiment': 'error',
                'confidence': 0.0
            }
    
    def get_popularity_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get popularity prediction using loaded model
        
        Args:
            features (Dict[str, Any]): Input features
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        if self.popularity_model is None or self.popularity_scaler is None or self.popularity_features is None:
            return {
                'error': 'Popularity model not loaded',
                'predicted_future_views': 0,
                'confidence_interval': '0-0'
            }
        
        try:
            # Prepare features in the correct order
            feature_values = []
            for feature in self.popularity_features:
                value = features.get(feature, 0)
                feature_values.append(value)
            
            # Scale features
            features_scaled = self.popularity_scaler.transform([feature_values])
            
            # Make prediction
            prediction = self.popularity_model.predict(features_scaled)[0]
            
            # Calculate confidence interval (simplified)
            prediction_lower = int(prediction * 0.8)
            prediction_upper = int(prediction * 1.2)
            
            return {
                'predicted_future_views': int(prediction),
                'prediction_lower_bound': prediction_lower,
                'prediction_upper_bound': prediction_upper,
                'confidence_interval': f"{prediction_lower}-{prediction_upper}",
                'features_used': self.popularity_features
            }
            
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'predicted_future_views': 0,
                'confidence_interval': '0-0'
            }
    
    def list_available_models(self) -> Dict[str, list]:
        """
        List all available model files
        
        Returns:
            Dict[str, list]: Dictionary of model types and their files
        """
        models = {
            'sentiment': [],
            'popularity': [],
            'other': []
        }
        
        # List all files in models directory
        for file_path in glob.glob(os.path.join(self.models_dir, '*')):
            file_name = os.path.basename(file_path)
            
            if 'sentiment' in file_name.lower():
                models['sentiment'].append(file_name)
            elif 'popularity' in file_name.lower():
                models['popularity'].append(file_name)
            else:
                models['other'].append(file_name)
        
        return models
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models
        
        Returns:
            Dict[str, Any]: Model information
        """
        info = {
            'sentiment_model_loaded': self.sentiment_model is not None,
            'popularity_model_loaded': self.popularity_model is not None,
            'models_directory': self.models_dir,
            'available_models': self.list_available_models()
        }
        
        if self.sentiment_model is not None:
            info['sentiment_model_type'] = type(self.sentiment_model).__name__
        
        if self.popularity_model is not None:
            info['popularity_model_type'] = type(self.popularity_model).__name__
            info['popularity_features_count'] = len(self.popularity_features) if self.popularity_features else 0
        
        return info


def main():
    """
    Test the model loader
    """
    print("=" * 60)
    print("MODEL LOADER TEST")
    print("=" * 60)
    
    # Initialize model loader
    loader = ModelLoader()
    
    # Load all models
    results = loader.load_all_models()
    
    # Get model info
    info = loader.get_model_info()
    print(f"\nModel Info: {info}")
    
    # Test sentiment prediction if model is loaded
    if results['sentiment']:
        print("\n" + "=" * 40)
        print("TESTING SENTIMENT PREDICTION")
        print("=" * 40)
        
        test_texts = [
            "I love this product! It's amazing!",
            "This is terrible and I hate it!",
            "It's okay, nothing special."
        ]
        
        for text in test_texts:
            result = loader.get_sentiment_prediction(text)
            print(f"\nText: {text}")
            print(f"Sentiment: {result.get('sentiment_name', 'error')}")
            print(f"Confidence: {result.get('confidence', 0):.3f}")
    
    # Test popularity prediction if model is loaded
    if results['popularity']:
        print("\n" + "=" * 40)
        print("TESTING POPULARITY PREDICTION")
        print("=" * 40)
        
        test_features = {
            'views': 10000,
            'likes': 500,
            'comments': 50,
            'engagement_rate': 5.5,
            'sentiment_score': 0.8
        }
        
        result = loader.get_popularity_prediction(test_features)
        print(f"\nFeatures: {test_features}")
        print(f"Predicted Future Views: {result.get('predicted_future_views', 0):,}")
        print(f"Confidence Interval: {result.get('confidence_interval', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("MODEL LOADER TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
