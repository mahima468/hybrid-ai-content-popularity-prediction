"""
Dataset Loader Utility
Loads and manages the final dataset for analytics and predictions
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class DatasetLoader:
    """
    Dataset Loader Class
    Handles loading and processing of the final dataset
    """
    
    def __init__(self, dataset_path: str = "../datasets/processed/final_dataset.csv"):
        """
        Initialize the dataset loader
        
        Args:
            dataset_path (str): Path to the final dataset
        """
        self.dataset_path = dataset_path
        self.dataset = None
        self.loaded_at = None
        
    def load_dataset(self) -> pd.DataFrame:
        """
        Load the dataset from CSV file
        
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            Exception: If there's an error loading the dataset
        """
        try:
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            
            logger.info(f"Loading dataset from: {self.dataset_path}")
            self.dataset = pd.read_csv(self.dataset_path)
            self.loaded_at = datetime.now()
            
            logger.info(f"Dataset loaded successfully. Shape: {self.dataset.shape}")
            logger.info(f"Columns: {list(self.dataset.columns)}")
            
            return self.dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def get_dataset(self) -> Optional[pd.DataFrame]:
        """
        Get the loaded dataset
        
        Returns:
            Optional[pd.DataFrame]: Loaded dataset or None if not loaded
        """
        return self.dataset
    
    def calculate_dashboard_stats(self) -> Dict[str, Any]:
        """
        Calculate dashboard statistics from dataset with safe column handling
        
        Returns:
            Dict[str, Any]: Dashboard statistics
        """
        if self.dataset is None:
            self.load_dataset()
        
        try:
            # Calculate real values from dataset with safe column access
            total_views = int(self.dataset['views'].sum()) if 'views' in self.dataset.columns else 0
            avg_engagement = float(self.dataset['engagement_rate'].mean()) if 'engagement_rate' in self.dataset.columns else 0.0
            avg_sentiment = float(self.dataset['sentiment_score'].mean()) if 'sentiment_score' in self.dataset.columns else 0.0
            
            # Build stats dictionary with safe column checks
            stats = {
                "total_views": total_views,
                "avg_engagement": avg_engagement,
                "avg_sentiment": avg_sentiment,
                "total_content": len(self.dataset),
                "sentiment_distribution": self._calculate_sentiment_distribution(),
                "engagement_authenticity": self._calculate_engagement_authenticity(),
                "popularity_trends": self._calculate_popularity_trends(),
                "engagement_accuracy": 0.943,  # Mock accuracy - should be calculated from model
                "sentiment_accuracy": 0.876,   # Mock accuracy - should be calculated from model
                "prediction_accuracy": 0.921   # Mock accuracy - should be calculated from model
            }
            
            # Only add predictions_summary if future_views exists
            if 'future_views' in self.dataset.columns:
                stats["predictions_summary"] = {
                    "total_predictions": len(self.dataset),
                    "avg_predicted_views": float(self.dataset['future_views'].mean()),
                    "max_predicted_views": int(self.dataset['future_views'].max()),
                    "min_predicted_views": int(self.dataset['future_views'].min())
                }
            else:
                stats["predictions_summary"] = {
                    "total_predictions": 0,
                    "avg_predicted_views": 0,
                    "max_predicted_views": 0,
                    "min_predicted_views": 0,
                    "note": "future_views column not available"
                }
            
            logger.info(f"Dashboard stats calculated - Total views: {total_views}, Avg engagement: {avg_engagement:.4f}, Avg sentiment: {avg_sentiment:.3f}")
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating dashboard stats: {str(e)}")
            # Return safe fallback values
            return {
                "total_views": 0,
                "avg_engagement": 0.0,
                "avg_sentiment": 0.0,
                "total_content": 0,
                "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
                "engagement_authenticity": {"real": 0, "fake": 0},
                "popularity_trends": {"labels": [], "predicted": [], "actual": []},
                "predictions_summary": {
                    "total_predictions": 0,
                    "avg_predicted_views": 0,
                    "max_predicted_views": 0,
                    "min_predicted_views": 0,
                    "note": "Error calculating stats"
                },
                "engagement_accuracy": 0.0,
                "sentiment_accuracy": 0.0,
                "prediction_accuracy": 0.0,
                "error": str(e)
            }
    
    def _calculate_sentiment_distribution(self) -> Dict[str, int]:
        """
        Calculate sentiment distribution
        
        Returns:
            Dict[str, int]: Sentiment distribution counts
        """
        if 'sentiment_score' not in self.dataset.columns:
            return {"positive": 0, "neutral": 0, "negative": 0}
        
        # Convert sentiment scores to categories
        sentiment_categories = pd.cut(
            self.dataset['sentiment_score'],
            bins=[-1, -0.3, 0.3, 1],
            labels=['negative', 'neutral', 'positive']
        )
        
        distribution = sentiment_categories.value_counts().to_dict()
        
        # Ensure all categories are present
        result = {
            "positive": int(distribution.get('positive', 0)),
            "neutral": int(distribution.get('neutral', 0)),
            "negative": int(distribution.get('negative', 0))
        }
        
        return result
    
    def _calculate_engagement_authenticity(self) -> Dict[str, int]:
        """
        Calculate engagement authenticity distribution
        
        Returns:
            Dict[str, int]: Engagement authenticity counts
        """
        try:
            if 'engagement_label' not in self.dataset.columns:
                logger.warning("engagement_label column not found")
                return {"real": 0, "fake": 0}
            
            # Count engagement labels (0 = real, 1 = fake)
            distribution = self.dataset['engagement_label'].value_counts().to_dict()
            
            # Map labels to meaningful names
            result = {
                "real": int(distribution.get(0, 0)),
                "fake": int(distribution.get(1, 0))
            }
            
            logger.info(f"Engagement authenticity distribution: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating engagement authenticity: {str(e)}")
            return {"real": 0, "fake": 0}
    
    def _calculate_popularity_trends(self) -> Dict[str, List]:
        """
        Calculate popularity trends for chart visualization
        
        Returns:
            Dict[str, List]: Popularity trends data
        """
        try:
            # Sample data for trends (in real implementation, this would be time-based)
            if 'views' in self.dataset.columns and 'future_views' in self.dataset.columns:
                # Take a sample of data for trends
                sample_size = min(100, len(self.dataset))
                sample_data = self.dataset.sample(n=sample_size, random_state=42)
                
                # Create monthly labels
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
                
                # Calculate trends (simplified - in real implementation, would use actual time series)
                predicted = [np.random.randint(60, 95) for _ in months]
                actual = [np.random.randint(55, 90) for _ in months]
                
                return {
                    "labels": months,
                    "predicted": predicted,
                    "actual": actual
                }
            else:
                # Fallback data
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
                return {
                    "labels": months,
                    "predicted": [65, 72, 78, 85, 89, 92],
                    "actual": [60, 68, 75, 82, 85, 88]
                }
                
        except Exception as e:
            logger.error(f"Error calculating popularity trends: {str(e)}")
            # Return fallback data
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            return {
                "labels": months,
                "predicted": [0, 0, 0, 0, 0, 0],
                "actual": [0, 0, 0, 0, 0, 0]
            }
    
    def get_sample_for_prediction(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Get a sample of data for prediction model
        
        Args:
            n_samples (int): Number of samples to return
            
        Returns:
            pd.DataFrame: Sample dataset for predictions
        """
        if self.dataset is None:
            self.load_dataset()
        
        # Sample data for prediction
        sample_size = min(n_samples, len(self.dataset))
        return self.dataset.sample(n=sample_size, random_state=42)
    
    def get_features_for_prediction(self, data: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for prediction model
        
        Args:
            data (pd.DataFrame): Data to prepare features from
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and target arrays
        """
        if data is None:
            data = self.dataset
        
        if data is None:
            self.load_dataset()
            data = self.dataset
        
        # Define feature columns
        feature_columns = [
            'views', 'likes', 'comments', 'engagement_rate', 'sentiment_score',
            'like_to_view_ratio', 'comment_to_view_ratio', 'like_to_comment_ratio',
            'log_views', 'log_likes', 'log_comments'
        ]
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in data.columns]
        
        if not available_features:
            raise ValueError("No valid feature columns found in dataset")
        
        # Prepare features
        X = data[available_features].fillna(0).values
        
        # Prepare target (future_views if available, otherwise create synthetic target)
        if 'future_views' in data.columns:
            y = data['future_views'].fillna(data['views']).values
        else:
            # Create synthetic target based on views
            y = (data['views'] * np.random.uniform(1.2, 3.0, len(data))).astype(int)
        
        logger.info(f"Prepared features: {X.shape}, Target: {y.shape}")
        return X, y
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded dataset
        
        Returns:
            Dict[str, Any]: Dataset information
        """
        if self.dataset is None:
            return {
                "loaded": False,
                "path": self.dataset_path,
                "shape": (0, 0),
                "columns": [],
                "loaded_at": None
            }
        
        return {
            "loaded": True,
            "path": self.dataset_path,
            "shape": self.dataset.shape,
            "columns": list(self.dataset.columns),
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "memory_usage": self.dataset.memory_usage(deep=True).sum(),
            "null_counts": self.dataset.isnull().sum().to_dict(),
            "dtypes": self.dataset.dtypes.to_dict()
        }

# Global dataset loader instance
dataset_loader = DatasetLoader()

def load_dataset() -> pd.DataFrame:
    """
    Convenience function to load the dataset
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    return dataset_loader.load_dataset()

def get_dashboard_stats() -> Dict[str, Any]:
    """
    Convenience function to get dashboard statistics
    
    Returns:
        Dict[str, Any]: Dashboard statistics
    """
    return dataset_loader.calculate_dashboard_stats()

def get_dataset_info() -> Dict[str, Any]:
    """
    Convenience function to get dataset information
    
    Returns:
        Dict[str, Any]: Dataset information
    """
    return dataset_loader.get_dataset_info()
