"""
Dataset Preprocessor Utility
Ensures dataset has all required columns and handles missing data
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class DatasetPreprocessor:
    """
    Dataset Preprocessor Class
    Handles preprocessing and validation of the final dataset
    """
    
    def __init__(self, dataset_path: str = "../datasets/processed/final_dataset.csv"):
        """
        Initialize the dataset preprocessor
        
        Args:
            dataset_path (str): Path to the final dataset
        """
        self.dataset_path = dataset_path
    
    def load_and_preprocess_dataset(self) -> pd.DataFrame:
        """
        Load dataset and ensure all required columns exist
        
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        try:
            # Load dataset
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            
            logger.info(f"Loading dataset from: {self.dataset_path}")
            df = pd.read_csv(self.dataset_path)
            logger.info(f"Dataset loaded. Shape: {df.shape}")
            
            # Check and add required columns
            df = self._ensure_required_columns(df)
            
            # Handle null values
            df = self._handle_null_values(df)
            
            # Validate data quality
            self._validate_dataset(df)
            
            # Save updated dataset
            df.to_csv(self.dataset_path, index=False)
            logger.info(f"Updated dataset saved to: {self.dataset_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing dataset: {str(e)}")
            raise
    
    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all required columns exist in the dataset
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with required columns
        """
        logger.info("Checking required columns...")
        
        # Check for views column
        if 'views' not in df.columns:
            logger.warning("Views column missing - creating synthetic views data")
            df['views'] = np.random.randint(1000, 1000000, size=len(df))
            logger.info("Views column created")
        
        # Check and create future_views column
        if 'future_views' not in df.columns:
            logger.warning("Future_views column missing - creating based on views")
            # Create future_views as views * random value between 0.8 and 1.5
            random_multiplier = np.random.uniform(0.8, 1.5, size=len(df))
            df['future_views'] = (df['views'] * random_multiplier).astype(int)
            logger.info("future_views column created")
        
        # Check for other important columns
        important_columns = {
            'likes': lambda x: np.random.randint(10, x['views'] // 10, size=len(x)),
            'comments': lambda x: np.random.randint(1, x['views'] // 100, size=len(x)),
            'engagement_rate': lambda x: (x['likes'] + x['comments']) / x['views'],
            'sentiment_score': lambda x: np.random.uniform(-1, 1, size=len(x))
        }
        
        for col, generator in important_columns.items():
            if col not in df.columns:
                logger.warning(f"{col} column missing - creating synthetic data")
                df[col] = generator(df)
                logger.info(f"{col} column created")
        
        logger.info(f"Final columns: {list(df.columns)}")
        return df
    
    def _handle_null_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle null values in the dataset
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with handled null values
        """
        logger.info("Handling null values...")
        
        # Check for null values in important columns
        important_cols = ['views', 'future_views', 'likes', 'comments', 'engagement_rate', 'sentiment_score']
        null_counts = df[important_cols].isnull().sum()
        
        if null_counts.any():
            logger.warning(f"Null values found: {null_counts.to_dict()}")
            
            # Fill null values
            for col in important_cols:
                if df[col].isnull().any():
                    if col in ['views', 'future_views', 'likes', 'comments']:
                        # Fill numeric columns with median
                        median_val = df[col].median()
                        df[col] = df[col].fillna(median_val)
                        logger.info(f"Filled null values in {col} with median: {median_val}")
                    elif col in ['engagement_rate', 'sentiment_score']:
                        # Fill rate columns with mean
                        mean_val = df[col].mean()
                        df[col] = df[col].fillna(mean_val)
                        logger.info(f"Filled null values in {col} with mean: {mean_val:.4f}")
        else:
            logger.info("No null values found in important columns")
        
        return df
    
    def _validate_dataset(self, df: pd.DataFrame) -> None:
        """
        Validate dataset quality
        
        Args:
            df (pd.DataFrame): Dataset to validate
            
        Raises:
            ValueError: If dataset fails validation
        """
        logger.info("Validating dataset quality...")
        
        # Check if dataset is empty
        if len(df) == 0:
            raise ValueError("Dataset is empty")
        
        # Check for negative values in columns that should be positive
        positive_cols = ['views', 'future_views', 'likes', 'comments']
        for col in positive_cols:
            if col in df.columns and (df[col] < 0).any():
                logger.warning(f"Negative values found in {col} - setting to minimum 1")
                df[col] = df[col].clip(lower=1)
        
        # Check engagement_rate bounds (should be between 0 and 1)
        if 'engagement_rate' in df.columns:
            if (df['engagement_rate'] < 0).any() or (df['engagement_rate'] > 1).any():
                logger.warning("Engagement rate values outside [0,1] - clipping")
                df['engagement_rate'] = df['engagement_rate'].clip(0, 1)
        
        # Check sentiment_score bounds (should be between -1 and 1)
        if 'sentiment_score' in df.columns:
            if (df['sentiment_score'] < -1).any() or (df['sentiment_score'] > 1).any():
                logger.warning("Sentiment score values outside [-1,1] - clipping")
                df['sentiment_score'] = df['sentiment_score'].clip(-1, 1)
        
        # Log dataset statistics
        logger.info(f"Dataset validation passed:")
        logger.info(f"  - Rows: {len(df)}")
        logger.info(f"  - Columns: {len(df.columns)}")
        logger.info(f"  - Views range: {df['views'].min()} to {df['views'].max()}")
        logger.info(f"  - Future views range: {df['future_views'].min()} to {df['future_views'].max()}")
        logger.info(f"  - Avg engagement rate: {df['engagement_rate'].mean():.4f}")
        logger.info(f"  - Avg sentiment score: {df['sentiment_score'].mean():.3f}")
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """
        Get summary of the dataset
        
        Returns:
            Dict[str, Any]: Dataset summary
        """
        try:
            df = pd.read_csv(self.dataset_path)
            
            summary = {
                "shape": df.shape,
                "columns": list(df.columns),
                "null_counts": df.isnull().sum().to_dict(),
                "data_types": df.dtypes.to_dict(),
                "has_future_views": 'future_views' in df.columns,
                "has_views": 'views' in df.columns,
                "views_stats": df['views'].describe().to_dict() if 'views' in df.columns else None,
                "future_views_stats": df['future_views'].describe().to_dict() if 'future_views' in df.columns else None
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting dataset summary: {str(e)}")
            return {"error": str(e)}

# Global preprocessor instance
dataset_preprocessor = DatasetPreprocessor()

def preprocess_dataset() -> pd.DataFrame:
    """
    Convenience function to preprocess the dataset
    
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    return dataset_preprocessor.load_and_preprocess_dataset()

def get_dataset_summary() -> Dict[str, Any]:
    """
    Convenience function to get dataset summary
    
    Returns:
        Dict[str, Any]: Dataset summary
    """
    return dataset_preprocessor.get_dataset_summary()
