"""
Dataset Enhancement Script
Adds fake engagement pattern detection to YouTube dataset
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_fake_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect fake engagement patterns based on specified rules
    
    Args:
        df (pd.DataFrame): Input dataframe with engagement metrics
        
    Returns:
        pd.DataFrame: Dataframe with engagement_label column added
    """
    logger.info("Applying fake engagement detection rules...")
    
    # Initialize engagement_label column (0 = real, 1 = fake)
    df['engagement_label'] = 0
    
    # Rule 1: If likes > 80% of views → suspicious
    rule1_mask = df['likes'] > (0.8 * df['views'])
    df.loc[rule1_mask, 'engagement_label'] = 1
    rule1_count = rule1_mask.sum()
    logger.info(f"Rule 1 - Likes > 80% of views: {rule1_count} suspicious videos")
    
    # Rule 2: If comments < 1% but likes very high → suspicious
    # Calculate 1% of views threshold
    comments_threshold = 0.01 * df['views']
    # Define "very high likes" as > 95th percentile
    likes_threshold = df['likes'].quantile(0.95)
    
    rule2_mask = (df['comments'] < comments_threshold) & (df['likes'] > likes_threshold)
    df.loc[rule2_mask, 'engagement_label'] = 1
    rule2_count = rule2_mask.sum()
    logger.info(f"Rule 2 - Comments < 1% but likes very high: {rule2_count} suspicious videos")
    logger.info(f"  Comments threshold: {comments_threshold.mean():.0f}, Likes threshold: {likes_threshold:.0f}")
    
    # Rule 3: If engagement_rate > 0.5 → suspicious
    rule3_mask = df['engagement_rate'] > 0.5
    df.loc[rule3_mask, 'engagement_label'] = 1
    rule3_count = rule3_mask.sum()
    logger.info(f"Rule 3 - Engagement rate > 0.5: {rule3_count} suspicious videos")
    
    # Combine all rules (if any rule matches, mark as suspicious)
    suspicious_mask = rule1_mask | rule2_mask | rule3_mask
    df.loc[suspicious_mask, 'engagement_label'] = 1
    
    # Ensure all real engagement stays as 0
    df.loc[~suspicious_mask, 'engagement_label'] = 0
    
    total_suspicious = suspicious_mask.sum()
    total_real = len(df) - total_suspicious
    
    logger.info(f"=== Fake Engagement Detection Summary ===")
    logger.info(f"Total videos analyzed: {len(df)}")
    logger.info(f"Suspicious engagement: {total_suspicious} ({total_suspicious/len(df)*100:.1f}%)")
    logger.info(f"Real engagement: {total_real} ({total_real/len(df)*100:.1f}%)")
    logger.info(f"Rule 1 violations: {rule1_count}")
    logger.info(f"Rule 2 violations: {rule2_count}")
    logger.info(f"Rule 3 violations: {rule3_count}")
    logger.info(f"Total unique suspicious: {total_suspicious}")
    
    return df

def add_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional engagement analysis features
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with additional engagement features
    """
    logger.info("Adding engagement analysis features...")
    
    # Additional features for better fake detection
    df['like_to_dislike_ratio'] = np.where(
        df['dislikes'] > 0,
        df['likes'] / df['dislikes'],
        df['likes']  # If no dislikes, ratio is just likes
    )
    
    # Log-transformed features for better distribution
    df['log_views'] = np.log1p(df['views'])
    df['log_likes'] = np.log1p(df['likes'])
    df['log_comments'] = np.log1p(df['comments'])
    
    # Engagement intensity score (0-1 scale)
    max_possible_engagement = df['likes'] + df['comments']
    df['engagement_intensity'] = max_possible_engagement / (df['views'] + 1)
    df['engagement_intensity'] = np.clip(df['engagement_intensity'], 0, 1)
    
    # Suspicious patterns based on statistical outliers
    df['views_outlier'] = np.abs(df['views'] - df['views'].median()) > (3 * df['views'].std())
    df['likes_outlier'] = np.abs(df['likes'] - df['likes'].median()) > (3 * df['likes'].std())
    df['comments_outlier'] = np.abs(df['comments'] - df['comments'].median()) > (3 * df['comments'].std())
    
    return df

def create_final_dataset(input_file: str, output_file: str) -> pd.DataFrame:
    """
    Create final enhanced dataset with fake engagement detection
    
    Args:
        input_file (str): Path to processed CSV file
        output_file (str): Path to save final enhanced CSV file
        
    Returns:
        pd.DataFrame: Final enhanced dataframe
    """
    logger.info(f"Loading processed dataset from: {input_file}")
    
    # Load the processed dataset
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Input dataset shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    
    # Add engagement analysis features
    df = add_engagement_features(df)
    
    # Apply fake engagement detection rules
    df = detect_fake_engagement(df)
    
    # Reorder columns for better readability
    column_order = [
        'views', 'likes', 'dislikes', 'comments',
        'engagement_rate', 'sentiment_score',
        'like_to_view_ratio', 'comment_to_view_ratio', 'like_to_comment_ratio',
        'like_to_dislike_ratio', 'log_views', 'log_likes', 'log_comments',
        'engagement_intensity', 'views_outlier', 'likes_outlier', 'comments_outlier',
        'engagement_label'
    ]
    
    # Ensure all columns exist
    for col in column_order:
        if col not in df.columns:
            df[col] = 0  # Default to 0 if missing
    
    df_final = df[column_order].copy()
    
    # Create summary statistics
    logger.info(f"Final dataset shape: {df_final.shape}")
    logger.info(f"Final columns: {df_final.columns.tolist()}")
    
    return df_final

def analyze_engagement_patterns(df: pd.DataFrame):
    """
    Analyze and report engagement patterns in the dataset
    
    Args:
        df (pd.DataFrame): Enhanced dataframe with engagement labels
    """
    logger.info("\n=== Engagement Pattern Analysis ===")
    
    # Overall statistics
    total_count = len(df)
    fake_count = df['engagement_label'].sum()
    real_count = total_count - fake_count
    
    logger.info(f"Total videos: {total_count}")
    logger.info(f"Fake engagement: {fake_count} ({fake_count/total_count*100:.1f}%)")
    logger.info(f"Real engagement: {real_count} ({real_count/total_count*100:.1f}%)")
    
    # Analyze by engagement rate ranges
    logger.info("\n--- Engagement Rate Analysis ---")
    rate_ranges = [
        (0, 0.01, "Very Low (0-1%)"),
        (0.01, 0.05, "Low (1-5%)"),
        (0.05, 0.15, "Normal (5-15%)"),
        (0.15, 0.30, "High (15-30%)"),
        (0.30, float('inf'), "Very High (>30%)")
    ]
    
    for i, (min_rate, max_rate, label) in enumerate(rate_ranges):
        mask = (df['engagement_rate'] >= min_rate) & (df['engagement_rate'] < max_rate)
        subset = df[mask]
        fake_in_range = subset['engagement_label'].sum()
        total_in_range = len(subset)
        
        if total_in_range > 0:
            fake_percentage = (fake_in_range / total_in_range) * 100
            logger.info(f"{label}: {fake_in_range}/{total_in_range} ({fake_percentage:.1f}% fake)")
    
    # Analyze suspicious patterns
    logger.info("\n--- Suspicious Pattern Analysis ---")
    suspicious_df = df[df['engagement_label'] == 1]
    
    if len(suspicious_df) > 0:
        logger.info(f"Suspicious videos stats:")
        logger.info(f"  Average views: {suspicious_df['views'].mean():.0f}")
        logger.info(f"  Average likes: {suspicious_df['likes'].mean():.0f}")
        logger.info(f"  Average comments: {suspicious_df['comments'].mean():.0f}")
        logger.info(f"  Average engagement rate: {suspicious_df['engagement_rate'].mean():.4f}")
        logger.info(f"  Average sentiment score: {suspicious_df['sentiment_score'].mean():.4f}")
    
    # Analyze real patterns
    logger.info("\n--- Real Pattern Analysis ---")
    real_df = df[df['engagement_label'] == 1]
    
    if len(real_df) > 0:
        logger.info(f"Real videos stats:")
        logger.info(f"  Average views: {real_df['views'].mean():.0f}")
        logger.info(f"  Average likes: {real_df['likes'].mean():.0f}")
        logger.info(f"  Average comments: {real_df['comments'].mean():.0f}")
        logger.info(f"  Average engagement rate: {real_df['engagement_rate'].mean():.4f}")
        logger.info(f"  Average sentiment score: {real_df['sentiment_score'].mean():.4f}")

def main():
    """Main enhancement function"""
    logger.info("=" * 60)
    logger.info("DATASET ENHANCEMENT - FAKE ENGAGEMENT DETECTION")
    logger.info("=" * 60)
    
    # Define input and output paths
    input_path = "datasets/processed/youtube_cleaned.csv"
    output_path = "datasets/processed/final_dataset.csv"
    
    try:
        # Check if input file exists
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            return
        
        # Create enhanced dataset
        enhanced_df = create_final_dataset(input_path, output_path)
        
        # Analyze engagement patterns
        analyze_engagement_patterns(enhanced_df)
        
        # Save enhanced dataset
        logger.info(f"Saving enhanced dataset to: {output_path}")
        enhanced_df.to_csv(output_path, index=False)
        
        # Create a smaller sample for testing
        sample_path = "datasets/processed/final_sample.csv"
        sample_df = enhanced_df.head(2000)  # First 2000 rows for testing
        sample_df.to_csv(sample_path, index=False)
        logger.info(f"Saved sample dataset to: {sample_path}")
        
        logger.info("=" * 60)
        logger.info("DATASET ENHANCEMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Files created:")
        logger.info(f"  - Final enhanced dataset: {output_path}")
        logger.info(f"  - Sample dataset: {sample_path}")
        logger.info(f"  - Total samples enhanced: {len(enhanced_df)}")
        logger.info(f"  - Fake engagement labeled: {enhanced_df['engagement_label'].sum()}")
        
    except Exception as e:
        logger.error(f"Error during enhancement: {e}")
        logger.error("=" * 60)
        logger.error("DATASET ENHANCEMENT FAILED")
        logger.error("=" * 60)

if __name__ == "__main__":
    main()
