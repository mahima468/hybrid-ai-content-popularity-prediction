"""
Engagement Detector Model
Uses IsolationForest to detect bot/fake engagement patterns
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import logging
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EngagementDetector:
    """
    Engagement Detector Class
    Detects suspicious/bot engagement patterns using IsolationForest
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize the engagement detector
        
        Args:
            contamination (float): Expected proportion of outliers (suspicious engagement)
            random_state (int): Random state for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_trained = False
        
        logger.info(f"Engagement detector initialized with contamination={contamination}")
    
    def generate_training_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """
        Generate synthetic training data for engagement detection
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Generated training data
        """
        logger.info(f"Generating {n_samples} synthetic engagement samples...")
        
        np.random.seed(self.random_state)
        
        data = []
        
        # Generate authentic engagement patterns (90% of data)
        authentic_samples = int(n_samples * 0.9)
        
        for i in range(authentic_samples):
            # Realistic engagement patterns
            views = np.random.lognormal(mean=8, sigma=1.5, size=1)[0]
            views = np.clip(views, 100, 1000000)
            
            # Realistic like-to-view ratio (1-5%)
            likes = int(views * np.random.uniform(0.01, 0.05))
            
            # Realistic comment-to-view ratio (0.1-1%)
            comments = int(views * np.random.uniform(0.001, 0.01))
            
            # Realistic share-to-view ratio (0.05-0.5%)
            shares = int(views * np.random.uniform(0.0005, 0.005))
            
            # Calculate engagement rate
            engagement_rate = ((likes + comments) / views) * 100 if views > 0 else 0
            
            data.append({
                'views': int(views),
                'likes': int(likes),
                'comments': int(comments),
                'shares': int(shares),
                'engagement_rate': engagement_rate,
                'label': 'authentic'
            })
        
        # Generate suspicious engagement patterns (10% of data)
        suspicious_samples = n_samples - authentic_samples
        
        for i in range(suspicious_samples):
            # Suspicious patterns
            views = np.random.lognormal(mean=7, sigma=1, size=1)[0]
            views = np.clip(views, 50, 500000)
            
            # Unusually high like-to-view ratio (10-30%)
            likes = int(views * np.random.uniform(0.1, 0.3))
            
            # Unusually high comment-to-view ratio (2-10%)
            comments = int(views * np.random.uniform(0.02, 0.1))
            
            # Unusually high share-to-view ratio (1-5%)
            shares = int(views * np.random.uniform(0.01, 0.05))
            
            # Calculate engagement rate
            engagement_rate = ((likes + comments) / views) * 100 if views > 0 else 0
            
            data.append({
                'views': int(views),
                'likes': int(likes),
                'comments': int(comments),
                'shares': int(shares),
                'engagement_rate': engagement_rate,
                'label': 'suspicious'
            })
        
        df = pd.DataFrame(data)
        
        # Add derived features
        df['like_to_view_ratio'] = df['likes'] / (df['views'] + 1)
        df['comment_to_view_ratio'] = df['comments'] / (df['views'] + 1)
        df['share_to_view_ratio'] = df['shares'] / (df['views'] + 1)
        df['like_to_comment_ratio'] = df['likes'] / (df['comments'] + 1)
        
        # Log transforms for skewed features
        df['log_views'] = np.log1p(df['views'])
        df['log_likes'] = np.log1p(df['likes'])
        df['log_comments'] = np.log1p(df['comments'])
        df['log_shares'] = np.log1p(df['shares'])
        
        logger.info(f"Generated dataset with {len(df)} samples")
        logger.info(f"Authentic samples: {len(df[df['label'] == 'authentic'])}")
        logger.info(f"Suspicious samples: {len(df[df['label'] == 'suspicious'])}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training/prediction
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Prepared features
        """
        # Define feature columns
        feature_columns = [
            'views', 'likes', 'comments', 'shares', 'engagement_rate',
            'like_to_view_ratio', 'comment_to_view_ratio', 'share_to_view_ratio',
            'like_to_comment_ratio', 'log_views', 'log_likes', 'log_comments', 'log_shares'
        ]
        
        # Ensure all features exist
        for col in feature_columns:
            if col not in df.columns:
                if col.startswith('log_'):
                    original_col = col.replace('log_', '')
                    if original_col in df.columns:
                        df[col] = np.log1p(df[original_col])
                    else:
                        df[col] = 0
                elif '_ratio' in col:
                    df[col] = 0
                else:
                    df[col] = 0
        
        # Handle missing values
        df[feature_columns] = df[feature_columns].fillna(0)
        
        self.feature_columns = feature_columns
        return df[feature_columns]
    
    def train(self, training_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Train the engagement detector
        
        Args:
            training_data (pd.DataFrame): Training data (optional)
            
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info("Training engagement detector...")
        
        # Generate training data if not provided
        if training_data is None:
            training_data = self.generate_training_data()
        
        # Prepare features
        X = self.prepare_features(training_data.copy())
        
        # Split data for evaluation
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=self.random_state)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto',
            max_features=1.0,
            bootstrap=False
        )
        
        self.model.fit(X_train_scaled)
        
        # Evaluate on test set
        y_pred_test = self.model.predict(X_test_scaled)
        y_scores_test = self.model.decision_function(X_test_scaled)
        
        # Calculate metrics
        outliers_test = (y_pred_test == -1).sum()
        inliers_test = (y_pred_test == 1).sum()
        
        # Approximate contamination rate
        actual_contamination = outliers_test / len(y_pred_test)
        
        self.is_trained = True
        
        results = {
            'model_type': 'IsolationForest',
            'contamination': self.contamination,
            'actual_contamination': actual_contamination,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'test_outliers': outliers_test,
            'test_inliers': inliers_test,
            'feature_columns': self.feature_columns
        }
        
        logger.info(f"✅ Engagement detector trained successfully")
        logger.info(f"   Expected contamination: {self.contamination:.2f}")
        logger.info(f"   Actual contamination: {actual_contamination:.2f}")
        logger.info(f"   Test outliers: {outliers_test}/{len(y_pred_test)}")
        
        return results
    
    def detect(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect engagement authenticity
        
        Args:
            features (Dict[str, Any]): Input features
            
        Returns:
            Dict[str, Any]: Detection results
        """
        if not self.is_trained:
            return {
                'error': 'Model not trained',
                'classification': 'unknown',
                'confidence': 0.0
            }
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([features])
            
            # Prepare features
            X = self.prepare_features(df)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            score = self.model.decision_function(X_scaled)[0]
            
            # Convert prediction to classification
            classification = 'suspicious' if prediction == -1 else 'authentic'
            
            # Calculate confidence based on score
            confidence = 1 / (1 + np.exp(-abs(score)))  # Sigmoid of absolute score
            
            return {
                'classification': classification,
                'is_suspicious': prediction == -1,
                'anomaly_score': float(score),
                'confidence': float(confidence),
                'features_used': self.feature_columns,
                'model_type': 'IsolationForest'
            }
            
        except Exception as e:
            logger.error(f"Error in engagement detection: {str(e)}")
            return {
                'error': f'Detection failed: {str(e)}',
                'classification': 'error',
                'confidence': 0.0
            }
    
    def batch_detect(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect engagement authenticity for multiple samples
        
        Args:
            features_list (List[Dict[str, Any]]): List of input features
            
        Returns:
            List[Dict[str, Any]]: List of detection results
        """
        results = []
        for features in features_list:
            result = self.detect(features)
            results.append(result)
        return results
    
    def save_model(self, save_dir: str = "saved_models") -> Dict[str, str]:
        """
        Save the trained model
        
        Args:
            save_dir (str): Directory to save the model
            
        Returns:
            Dict[str, str]: Paths to saved files
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = os.path.join(save_dir, f"engagement_detector_{timestamp}.joblib")
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(save_dir, f"engagement_detector_scaler_{timestamp}.joblib")
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'model_type': 'IsolationForest',
            'contamination': self.contamination,
            'feature_columns': self.feature_columns,
            'timestamp': timestamp,
            'model_path': model_path,
            'scaler_path': scaler_path
        }
        
        import json
        metadata_path = os.path.join(save_dir, f"engagement_detector_metadata_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Engagement detector saved to {save_dir}")
        
        return {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'metadata_path': metadata_path
        }
    
    def load_model(self, model_path: str, scaler_path: str) -> bool:
        """
        Load a trained model
        
        Args:
            model_path (str): Path to model file
            scaler_path (str): Path to scaler file
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            # Load metadata if available
            metadata_path = model_path.replace('.joblib', '_metadata.json')
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.feature_columns = metadata.get('feature_columns', [])
                self.contamination = metadata.get('contamination', 0.1)
            
            self.is_trained = True
            logger.info(f"✅ Engagement detector loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading engagement detector: {str(e)}")
            return False

# Global instance
engagement_detector = EngagementDetector()

def get_engagement_detector() -> EngagementDetector:
    """Get global engagement detector instance"""
    return engagement_detector
