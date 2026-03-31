"""
Data Preprocessing Script
Processes YouTube dataset for Hybrid AI Content Popularity Prediction System
"""

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_youtube_data(input_file: str, output_file: str) -> pd.DataFrame:
    """
    Preprocess YouTube dataset
    
    Args:
        input_file (str): Path to raw CSV file
        output_file (str): Path to save processed CSV file
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    logger.info(f"Loading dataset from: {input_file}")
    
    # Load the dataset
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Original dataset shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    
    # Select useful columns
    required_columns = ['views', 'likes', 'dislikes', 'comment_count']
    available_columns = [col for col in required_columns if col in df.columns]
    
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        logger.error(f"Missing required columns: {missing_cols}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
    
    logger.info(f"Using columns: {available_columns}")
    
    # Create a copy with only required columns
    df_processed = df[available_columns].copy()
    
    # Remove null values
    initial_rows = len(df_processed)
    df_processed = df_processed.dropna()
    removed_rows = initial_rows - len(df_processed)
    logger.info(f"Removed {removed_rows} rows with null values")
    logger.info(f"Dataset shape after removing nulls: {df_processed.shape}")
    
    # Convert columns to numeric, handling any string values
    for col in available_columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Remove any remaining null values after conversion
    df_processed = df_processed.dropna()
    logger.info(f"Dataset shape after numeric conversion: {df_processed.shape}")
    
    # Create engagement_rate feature
    df_processed['engagement_rate'] = (df_processed['likes'] + df_processed['comment_count']) / df_processed['views']
    df_processed['engagement_rate'] = df_processed['engagement_rate'].fillna(0)
    
    # Handle division by zero (where views = 0)
    df_processed.loc[df_processed['views'] == 0, 'engagement_rate'] = 0
    
    # Create temporary sentiment_score (random between -1 and 1)
    # In a real scenario, this would come from sentiment analysis
    np.random.seed(42)  # For reproducibility
    df_processed['sentiment_score'] = np.random.uniform(-1, 1, size=len(df_processed))
    
    # Normalize column names
    df_processed = df_processed.rename(columns={
        'comment_count': 'comments'
    })
    
    # Create additional useful features
    df_processed['like_to_view_ratio'] = df_processed['likes'] / (df_processed['views'] + 1)
    df_processed['comment_to_view_ratio'] = df_processed['comments'] / (df_processed['views'] + 1)
    df_processed['like_to_comment_ratio'] = df_processed['likes'] / (df_processed['comments'] + 1)
    
    # Handle potential infinity values
    df_processed = df_processed.replace([np.inf, -np.inf], 0)
    
    # Select final columns for output
    output_columns = [
        'views', 'likes', 'dislikes', 'comments', 
        'engagement_rate', 'sentiment_score',
        'like_to_view_ratio', 'comment_to_view_ratio', 'like_to_comment_ratio'
    ]
    
    df_final = df_processed[output_columns].copy()
    
    # Log some statistics
    logger.info("\n=== Dataset Statistics ===")
    logger.info(f"Total samples: {len(df_final)}")
    logger.info(f"Views - Mean: {df_final['views'].mean():.0f}, Std: {df_final['views'].std():.0f}")
    logger.info(f"Likes - Mean: {df_final['likes'].mean():.0f}, Std: {df_final['likes'].std():.0f}")
    logger.info(f"Comments - Mean: {df_final['comments'].mean():.0f}, Std: {df_final['comments'].std():.0f}")
    logger.info(f"Engagement Rate - Mean: {df_final['engagement_rate'].mean():.4f}, Std: {df_final['engagement_rate'].std():.4f}")
    logger.info(f"Sentiment Score - Mean: {df_final['sentiment_score'].mean():.4f}, Std: {df_final['sentiment_score'].std():.4f}")
    
    return df_final

def main():
    """Main preprocessing function"""
    logger.info("=" * 60)
    logger.info("YOUTUBE DATASET PREPROCESSING")
    logger.info("=" * 60)
    
    # Define input and output paths
    input_path = "datasets/raw/archive/USvideos.csv"
    output_path = "datasets/processed/youtube_cleaned.csv"
    
    try:
        # Check if input file exists
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            return
        
        # Preprocess the data
        processed_df = preprocess_youtube_data(input_path, output_path)
        
        # Save processed dataset
        logger.info(f"Saving processed dataset to: {output_path}")
        processed_df.to_csv(output_path, index=False)
        
        # Save a smaller sample for quick testing
        sample_path = "datasets/processed/youtube_sample.csv"
        sample_df = processed_df.head(1000)  # First 1000 rows
        sample_df.to_csv(sample_path, index=False)
        logger.info(f"Saved sample dataset to: {sample_path}")
        
        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Files created:")
        logger.info(f"  - Full processed dataset: {output_path}")
        logger.info(f"  - Sample dataset: {sample_path}")
        logger.info(f"  - Total samples processed: {len(processed_df)}")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        logger.error("=" * 60)
        logger.error("PREPROCESSING FAILED")
        logger.error("=" * 60)

if __name__ == "__main__":
    main()
