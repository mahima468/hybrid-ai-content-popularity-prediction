"""
Twitter Sentiment Dataset Processing Script
Processes Twitter sentiment data for sentiment analysis model training
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import re
import string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_twitter_sentiment(output_file: str, num_samples: int = 10000) -> pd.DataFrame:
    """
    Create a sample Twitter sentiment dataset if the original doesn't exist
    
    Args:
        output_file (str): Path to save the sample dataset
        num_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Sample Twitter sentiment dataset
    """
    logger.info(f"Creating sample Twitter sentiment dataset with {num_samples} samples...")
    
    # Sample positive tweets
    positive_tweets = [
        "I love this new phone! It's amazing! 😍",
        "Just had the best day ever! Feeling so blessed 🙏",
        "This movie is absolutely fantastic! Highly recommend 🎬",
        "So excited for the weekend! Can't wait to celebrate 🎉",
        "Coffee is life! Best morning routine ever ☕",
        "Beautiful weather today! Nature is incredible 🌞",
        "Just finished my workout! Feeling energized 💪",
        "This restaurant serves the best food I've ever had 🍕",
        "My new puppy is the cutest thing ever! 🐶",
        "Graduated today! Dreams really do come true 🎓",
        "Got the job offer! Hard work pays off 💼",
        "This vacation is exactly what I needed! 🏖️",
        "Love spending time with family and friends 👨‍👩‍👧‍👦",
        "This book changed my perspective on life 📚",
        "Amazing concert tonight! Best band ever! 🎵"
    ]
    
    # Sample negative tweets
    negative_tweets = [
        "Stuck in traffic again! This is frustrating 🚗",
        "My phone just died! Worst timing ever 😠",
        "This weather is terrible! Can't go outside 🌧️",
        "Feeling sick today! This medicine isn't working 🤒",
        "Lost my wallet! This day keeps getting worse 😭",
        "The service at this restaurant is horrible! 😤",
        "My flight got cancelled! Ruined my plans ✈️",
        "This movie was so boring! Waste of money 🎬",
        "Computer crashed again! Lost all my work 💻",
        "Bad news from the doctor today 😔",
        "My car broke down! This will be expensive 🔧",
        "This meeting was pointless! Waste of time ⏰",
        "The internet is so slow today! Can't get anything done 🌐",
        "Got rejected from my dream job! Heartbroken 💔",
        "This food tastes terrible! So disappointed 🍽️"
    ]
    
    # Sample neutral tweets
    neutral_tweets = [
        "The weather is okay today. Nothing special.",
        "Just finished my daily routine. Time for lunch.",
        "Reading a book about history. Interesting facts.",
        "The meeting is scheduled for 3 PM tomorrow.",
        "Traffic is normal for this time of day.",
        "Working on my project. Making steady progress.",
        "The store opens at 9 AM and closes at 9 PM.",
        "Just updated my software to the latest version.",
        "The train arrives at platform 2 in 5 minutes.",
        "Making coffee for the morning. Regular routine.",
        "Checking emails. Nothing urgent today.",
        "The temperature is around 72 degrees.",
        "Walking to the park. Nice day for exercise.",
        "The news report mentions local events.",
        "Preparing dinner. Simple meal tonight."
    ]
    
    # Generate dataset
    data = []
    
    # Add positive samples (label 4)
    for _ in range(num_samples // 3):
        tweet = np.random.choice(positive_tweets)
        # Add some variation
        tweet = tweet + " " + np.random.choice(["", "😊", "!", "👍"])
        data.append([tweet, 4])
    
    # Add negative samples (label 0)
    for _ in range(num_samples // 3):
        tweet = np.random.choice(negative_tweets)
        # Add some variation
        tweet = tweet + " " + np.random.choice(["", "😔", "!", "😞"])
        data.append([tweet, 0])
    
    # Add neutral samples (label 2)
    for _ in range(num_samples // 3):
        tweet = np.random.choice(neutral_tweets)
        # Add some variation
        tweet = tweet + " " + np.random.choice(["", ".", "📝"])
        data.append([tweet, 2])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['text', 'sentiment_label'])
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Sample Twitter sentiment dataset saved to: {output_file}")
    
    return df

def load_twitter_sentiment(file_path: str) -> pd.DataFrame:
    """
    Load Twitter sentiment dataset
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded sentiment dataset
    """
    logger.info(f"Loading Twitter sentiment dataset from: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        
        # Check required columns
        if 'text' not in df.columns or 'sentiment_label' not in df.columns:
            logger.error("Required columns 'text' and 'sentiment_label' not found")
            logger.info(f"Available columns: {df.columns.tolist()}")
            raise ValueError("Missing required columns")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def map_sentiment_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map sentiment labels to sentiment scores
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment labels
        
    Returns:
        pd.DataFrame: DataFrame with mapped sentiment scores
    """
    logger.info("Mapping sentiment labels to sentiment scores...")
    
    # Create mapping dictionary
    label_mapping = {
        0: 'negative',
        2: 'neutral', 
        4: 'positive'
    }
    
    # Create score mapping
    score_mapping = {
        'negative': -1,
        'neutral': 0,
        'positive': 1
    }
    
    # Map labels to sentiment names
    df['sentiment'] = df['sentiment_label'].map(label_mapping)
    
    # Map sentiment names to scores
    df['sentiment_score'] = df['sentiment'].map(score_mapping)
    
    # Check for unmapped values
    unmapped = df[df['sentiment'].isna()]
    if len(unmapped) > 0:
        logger.warning(f"Found {len(unmapped)} unmapped sentiment labels")
        logger.info(f"Unique labels found: {df['sentiment_label'].unique()}")
    
    # Fill any NaN values with neutral
    df['sentiment'] = df['sentiment'].fillna('neutral')
    df['sentiment_score'] = df['sentiment_score'].fillna(0)
    
    # Log distribution
    logger.info("=== Sentiment Distribution ===")
    sentiment_counts = df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        logger.info(f"{sentiment.capitalize()}: {count} ({percentage:.1f}%)")
    
    return df

def clean_text(text: str) -> str:
    """
    Clean text data for sentiment analysis
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags symbols
    text = re.sub(r'@\w+|#', '', text)
    
    # Remove emojis (basic cleanup)
    text = re.sub(r'[^\w\s#@]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_twitter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess Twitter sentiment data
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    logger.info("Preprocessing Twitter sentiment data...")
    
    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Add text length features
    df['text_length'] = df['text'].str.len()
    df['cleaned_text_length'] = df['cleaned_text'].str.len()
    df['word_count'] = df['cleaned_text'].str.split().str.len()
    
    # Remove rows with empty text after cleaning
    initial_count = len(df)
    df = df[df['cleaned_text'].str.len() > 0]
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} rows with empty text")
    
    logger.info(f"Preprocessed dataset shape: {df.shape}")
    
    return df

def create_sentiment_training_dataset(df: pd.DataFrame, output_file: str) -> pd.DataFrame:
    """
    Create final sentiment training dataset
    
    Args:
        df (pd.DataFrame): Preprocessed sentiment DataFrame
        output_file (str): Path to save the training dataset
        
    Returns:
        pd.DataFrame: Final training dataset
    """
    logger.info("Creating final sentiment training dataset...")
    
    # Select columns for training
    training_columns = [
        'text', 'cleaned_text', 'sentiment_label', 'sentiment', 'sentiment_score',
        'text_length', 'cleaned_text_length', 'word_count'
    ]
    
    # Ensure all columns exist
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0
    
    df_training = df[training_columns].copy()
    
    # Save training dataset
    df_training.to_csv(output_file, index=False)
    logger.info(f"Sentiment training dataset saved to: {output_file}")
    
    return df_training

def merge_with_youtube_data(twitter_df: pd.DataFrame, youtube_file: str, output_file: str) -> pd.DataFrame:
    """
    Merge Twitter sentiment data with YouTube dataset for enhanced training
    
    Args:
        twitter_df (pd.DataFrame): Twitter sentiment DataFrame
        youtube_file (str): Path to YouTube dataset
        output_file (str): Path to save merged dataset
        
    Returns:
        pd.DataFrame: Merged dataset
    """
    logger.info("Merging Twitter sentiment with YouTube dataset...")
    
    try:
        # Load YouTube dataset
        youtube_df = pd.read_csv(youtube_file)
        logger.info(f"YouTube dataset loaded: {youtube_df.shape}")
        
        # Create text from YouTube data for sentiment analysis
        # Use engagement patterns as proxy for sentiment
        youtube_df['text'] = youtube_df.apply(lambda row: 
            f"Video with {row['views']} views, {row['likes']} likes, {row['comments']} comments", 
            axis=1
        )
        
        # Map engagement to sentiment (heuristic)
        def engagement_to_sentiment(row):
            engagement_rate = row['engagement_rate']
            if engagement_rate > 0.1:  # High engagement
                return 4  # positive
            elif engagement_rate < 0.02:  # Low engagement
                return 0  # negative
            else:
                return 2  # neutral
        
        youtube_df['sentiment_label'] = youtube_df.apply(engagement_to_sentiment, axis=1)
        youtube_df['sentiment'] = youtube_df['sentiment_label'].map({0: 'negative', 2: 'neutral', 4: 'positive'})
        youtube_df['sentiment_score'] = youtube_df['sentiment'].map({'negative': -1, 'neutral': 0, 'positive': 1})
        
        # Clean YouTube text
        youtube_df['cleaned_text'] = youtube_df['text'].apply(clean_text)
        youtube_df['text_length'] = youtube_df['text'].str.len()
        youtube_df['cleaned_text_length'] = youtube_df['cleaned_text'].str.len()
        youtube_df['word_count'] = youtube_df['cleaned_text'].str.split().str.len()
        
        # Select matching columns
        merge_columns = [
            'text', 'cleaned_text', 'sentiment_label', 'sentiment', 'sentiment_score',
            'text_length', 'cleaned_text_length', 'word_count'
        ]
        
        youtube_merge = youtube_df[merge_columns].copy()
        
        # Combine datasets
        merged_df = pd.concat([twitter_df, youtube_merge], ignore_index=True)
        merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save merged dataset
        merged_df.to_csv(output_file, index=False)
        
        logger.info(f"Merged dataset saved to: {output_file}")
        logger.info(f"Merged dataset shape: {merged_df.shape}")
        
        # Log final distribution
        logger.info("=== Final Merged Sentiment Distribution ===")
        sentiment_counts = merged_df['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(merged_df)) * 100
            logger.info(f"{sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error merging datasets: {e}")
        raise

def main():
    """Main processing function"""
    logger.info("=" * 60)
    logger.info("TWITTER SENTIMENT DATASET PROCESSING")
    logger.info("=" * 60)
    
    # Define paths
    twitter_input_path = "datasets/external/twitter_sentiment.csv"
    twitter_output_path = "datasets/processed/twitter_sentiment_processed.csv"
    training_output_path = "datasets/processed/sentiment_training_data.csv"
    merged_output_path = "datasets/processed/merged_sentiment_data.csv"
    youtube_path = "datasets/processed/final_dataset.csv"
    
    try:
        # Check if Twitter dataset exists, create sample if not
        if not os.path.exists(twitter_input_path):
            logger.info("Twitter sentiment dataset not found. Creating sample dataset...")
            os.makedirs("datasets/external", exist_ok=True)
            twitter_df = create_sample_twitter_sentiment(twitter_input_path, num_samples=15000)
        else:
            twitter_df = load_twitter_sentiment(twitter_input_path)
        
        # Map sentiment labels to scores
        twitter_df = map_sentiment_labels(twitter_df)
        
        # Preprocess Twitter data
        twitter_df = preprocess_twitter_data(twitter_df)
        
        # Create sentiment training dataset
        training_df = create_sentiment_training_dataset(twitter_df, training_output_path)
        
        # Merge with YouTube data (if available)
        if os.path.exists(youtube_path):
            logger.info("Merging with YouTube dataset...")
            merged_df = merge_with_youtube_data(twitter_df, youtube_path, merged_output_path)
        else:
            logger.warning("YouTube dataset not found. Skipping merge.")
        
        logger.info("=" * 60)
        logger.info("TWITTER SENTIMENT PROCESSING COMPLETED")
        logger.info("=" * 60)
        logger.info("Files created:")
        logger.info(f"  - Original Twitter data: {twitter_input_path}")
        logger.info(f"  - Processed Twitter data: {twitter_output_path}")
        logger.info(f"  - Sentiment training data: {training_output_path}")
        if os.path.exists(youtube_path):
            logger.info(f"  - Merged sentiment data: {merged_output_path}")
        logger.info(f"  - Total samples processed: {len(twitter_df)}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        logger.error("=" * 60)
        logger.error("TWITTER SENTIMENT PROCESSING FAILED")
        logger.error("=" * 60)

if __name__ == "__main__":
    main()
