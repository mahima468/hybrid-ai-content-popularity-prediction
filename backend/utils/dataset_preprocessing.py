"""
Dataset Preprocessing Utilities
Specialized preprocessing functions for IMDB movie reviews and YouTube trending datasets
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings
from urllib.parse import urlparse
import json

warnings.filterwarnings('ignore')

class DatasetPreprocessor:
    """
    Dataset Preprocessing Class
    Specialized functions for loading and preprocessing IMDB and YouTube datasets
    """
    
    def __init__(self):
        """Initialize the preprocessor with NLTK components"""
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Download required NLTK data
        try:
            self.stop_words = set(stopwords.words('english'))
            nltk.data.find('corpora/stopwords')
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str, remove_stopwords: bool = True, 
                   lemmatize: bool = True, min_length: int = 3,
                   remove_html: bool = True, remove_urls: bool = True) -> str:
        """
        Clean and preprocess text data for sentiment analysis
        
        Args:
            text (str): Input text to clean
            remove_stopwords (bool): Whether to remove stopwords
            lemmatize (bool): Whether to lemmatize words
            min_length (int): Minimum word length to keep
            remove_html (bool): Whether to remove HTML tags
            remove_urls (bool): Whether to remove URLs
            
        Returns:
            str: Cleaned and preprocessed text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags (common in IMDB reviews)
        if remove_html:
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'&[a-z]+;', '', text)  # Remove HTML entities
        
        # Remove URLs
        if remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and numbers (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Remove stopwords if requested
            if remove_stopwords and token in self.stop_words:
                continue
            
            # Filter by length
            if len(token) < min_length:
                continue
            
            # Lemmatize if requested
            if lemmatize:
                try:
                    token = self.lemmatizer.lemmatize(token)
                except:
                    pass
            
            filtered_tokens.append(token)
        
        return ' '.join(filtered_tokens)
    
    def load_imdb_dataset(self, file_path: str = None, sample_size: int = None,
                         download_sample: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load and preprocess IMDB movie reviews dataset
        
        Args:
            file_path (str): Path to IMDB dataset CSV file
            sample_size (int): Number of samples to load (for testing)
            download_sample (bool): Whether to download a sample dataset
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Processed dataset and metadata
        """
        if download_sample or file_path is None:
            # Generate sample IMDB-style dataset
            print("Generating sample IMDB dataset...")
            df = self._generate_sample_imdb_data(sample_size or 5000)
            metadata = {
                'source': 'generated_sample',
                'total_samples': len(df),
                'columns': list(df.columns),
                'positive_reviews': len(df[df['sentiment'] == 'positive']),
                'negative_reviews': len(df[df['sentiment'] == 'negative'])
            }
        else:
            # Load from file
            try:
                df = pd.read_csv(file_path)
                print(f"Loaded IMDB dataset from {file_path}")
                
                # Standardize column names
                if 'review' in df.columns:
                    df.rename(columns={'review': 'text'}, inplace=True)
                elif 'text' not in df.columns:
                    raise ValueError("Dataset must contain 'review' or 'text' column")
                
                # Ensure sentiment column exists
                if 'sentiment' not in df.columns:
                    # Try to infer sentiment from other columns
                    if 'rating' in df.columns:
                        df['sentiment'] = df['rating'].apply(
                            lambda x: 'positive' if x >= 7 else 'negative' if x <= 4 else 'neutral'
                        )
                    else:
                        raise ValueError("Dataset must contain 'sentiment' or 'rating' column")
                
                # Sample if requested
                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                
                metadata = {
                    'source': file_path,
                    'total_samples': len(df),
                    'columns': list(df.columns),
                    'positive_reviews': len(df[df['sentiment'] == 'positive']),
                    'negative_reviews': len(df[df['sentiment'] == 'negative'])
                }
                
            except Exception as e:
                raise ValueError(f"Error loading IMDB dataset: {str(e)}")
        
        # Clean text data
        print("Cleaning text data...")
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Add text features
        df['text_length'] = df['text'].str.len()
        df['cleaned_length'] = df['cleaned_text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        
        # Add sentiment scores using VADER
        print("Calculating sentiment scores...")
        sentiment_scores = df['text'].apply(self.sentiment_analyzer.polarity_scores)
        df['sentiment_score'] = [score['compound'] for score in sentiment_scores]
        df['sentiment_pos'] = [score['pos'] for score in sentiment_scores]
        df['sentiment_neg'] = [score['neg'] for score in sentiment_scores]
        df['sentiment_neu'] = [score['neu'] for score in sentiment_scores]
        
        # Convert sentiment to numeric
        sentiment_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
        df['sentiment_label'] = df['sentiment'].map(sentiment_mapping)
        
        print(f"Processed {len(df)} IMDB reviews")
        return df, metadata
    
    def _generate_sample_imdb_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate sample IMDB-style movie reviews"""
        np.random.seed(42)
        
        positive_reviews = [
            "This movie was absolutely fantastic! Great acting and storyline.",
            "Amazing film! Would definitely watch again.",
            "Brilliant performance by the lead actor. Highly recommended.",
            "Excellent cinematography and direction. A must-watch!",
            "Incredible movie with great character development.",
            "Outstanding film! The best I've seen this year.",
            "Wonderful storytelling and amazing visuals.",
            "Perfect movie! Couldn't find any flaws.",
            "Spectacular film with great emotional depth.",
            "Marvelous movie that exceeded all expectations.",
            "A true masterpiece of cinema. Absolutely brilliant!",
            "Engaging from start to finish. Great chemistry between actors.",
            "Beautifully shot and well-written. A cinematic gem.",
            "Powerful performances and compelling narrative.",
            "One of the best films I've ever seen. Truly exceptional!"
        ]
        
        negative_reviews = [
            "Terrible movie! Waste of time and money.",
            "Awful film with poor acting and weak plot.",
            "Disappointing movie. Not worth watching.",
            "Bad movie with boring storyline.",
            "Horrible film! Couldn't even finish it.",
            "Poor movie with terrible direction.",
            "Worst movie I've ever seen. Avoid at all costs.",
            "Dreadful film with no redeeming qualities.",
            "Awful movie with terrible dialogue.",
            "Terrible experience. Complete waste of time.",
            "Boring and predictable. Skip this one.",
            "Poorly executed and badly acted. A disaster.",
            "Uninteresting plot with wooden performances.",
            "Completely failed to deliver on its premise.",
            "A mess of a movie. Don't waste your money."
        ]
        
        neutral_reviews = [
            "The movie was okay. Not great, not terrible.",
            "Average film with some good moments.",
            "Decent movie but could have been better.",
            "It was fine. Nothing special.",
            "The film had its ups and downs.",
            "Mixed feelings about this one.",
            "Somewhat entertaining but forgettable.",
            "An average experience overall.",
            "Neither good nor bad, just mediocre.",
            "The movie was watchable but unremarkable."
        ]
        
        # Generate reviews
        reviews = []
        sentiments = []
        
        for i in range(n_samples):
            sentiment_choice = np.random.random()
            
            if sentiment_choice < 0.4:  # 40% positive
                base_review = np.random.choice(positive_reviews)
                sentiment = 'positive'
            elif sentiment_choice < 0.8:  # 40% negative
                base_review = np.random.choice(negative_reviews)
                sentiment = 'negative'
            else:  # 20% neutral
                base_review = np.random.choice(neutral_reviews)
                sentiment = 'neutral'
            
            # Add some variation
            if np.random.random() > 0.7:
                extra_text = np.random.choice([
                    " Really enjoyed it!",
                    " Could be better.",
                    " Worth watching once.",
                    " Not my cup of tea.",
                    " Surprisingly good!",
                    " Quite disappointing."
                ])
                review = base_review + extra_text
            else:
                review = base_review
            
            reviews.append(review)
            sentiments.append(sentiment)
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': reviews,
            'sentiment': sentiments,
            'rating': np.random.randint(1, 11, n_samples)  # 1-10 rating
        })
        
        return df
    
    def load_youtube_dataset(self, file_path: str = None, sample_size: int = None,
                           download_sample: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load and preprocess YouTube trending dataset
        
        Args:
            file_path (str): Path to YouTube dataset CSV file
            sample_size (int): Number of samples to load (for testing)
            download_sample (bool): Whether to download a sample dataset
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Processed dataset and metadata
        """
        if download_sample or file_path is None:
            # Generate sample YouTube-style dataset
            print("Generating sample YouTube dataset...")
            df = self._generate_sample_youtube_data(sample_size or 10000)
            metadata = {
                'source': 'generated_sample',
                'total_samples': len(df),
                'columns': list(df.columns),
                'categories': df['category'].unique().tolist(),
                'date_range': (df['trending_date'].min(), df['trending_date'].max())
            }
        else:
            # Load from file
            try:
                df = pd.read_csv(file_path)
                print(f"Loaded YouTube dataset from {file_path}")
                
                # Standardize common column names
                column_mapping = {
                    'video_id': 'video_id',
                    'title': 'title',
                    'channel_title': 'channel_title',
                    'category_id': 'category_id',
                    'tags': 'tags',
                    'views': 'views',
                    'likes': 'likes',
                    'dislikes': 'dislikes',
                    'comment_count': 'comment_count',
                    'thumbnail_link': 'thumbnail_link',
                    'date': 'trending_date'
                }
                
                # Apply column mapping if needed
                for old_col, new_col in column_mapping.items():
                    if old_col in df.columns and new_col not in df.columns:
                        df.rename(columns={old_col: new_col}, inplace=True)
                
                # Sample if requested
                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                
                metadata = {
                    'source': file_path,
                    'total_samples': len(df),
                    'columns': list(df.columns),
                    'categories': df['category'].unique().tolist() if 'category' in df.columns else [],
                    'date_range': (df['trending_date'].min(), df['trending_date'].max()) if 'trending_date' in df.columns else None
                }
                
            except Exception as e:
                raise ValueError(f"Error loading YouTube dataset: {str(e)}")
        
        # Process the dataset
        df = self._process_youtube_data(df)
        
        print(f"Processed {len(df)} YouTube trending videos")
        return df, metadata
    
    def _generate_sample_youtube_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate sample YouTube-style trending data"""
        np.random.seed(42)
        
        categories = ['Entertainment', 'Music', 'Gaming', 'News & Politics', 'Sports', 
                     'Comedy', 'Education', 'Science & Technology', 'Film & Animation', 'Howto & Style']
        
        # Generate video data
        data = []
        for i in range(n_samples):
            # Base metrics
            views = np.random.lognormal(mean=10, sigma=2, size=1)[0].astype(int)
            views = np.clip(views, 1000, 100000000)
            
            # Engagement metrics (correlated with views)
            like_rate = np.random.beta(2, 20, size=1)[0]  # Average like rate ~10%
            likes = int(views * like_rate)
            
            dislike_rate = np.random.beta(1, 50, size=1)[0]  # Average dislike rate ~2%
            dislikes = int(views * dislike_rate)
            
            comment_rate = np.random.beta(1, 100, size=1)[0]  # Average comment rate ~1%
            comments = int(views * comment_rate)
            
            # Channel metrics
            channel_subscribers = np.random.lognormal(mean=8, sigma=2, size=1)[0].astype(int)
            channel_subscribers = np.clip(channel_subscribers, 1000, 50000000)
            
            # Time-based features
            trending_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365))
            publish_date = trending_date - timedelta(days=np.random.randint(1, 30))
            
            data.append({
                'video_id': f'video_{i:08d}',
                'title': f'Trending Video #{i+1}',
                'channel_title': f'Channel {np.random.randint(1, 1000)}',
                'category': np.random.choice(categories),
                'tags': f'tag{i},trending,viral',
                'views': views,
                'likes': likes,
                'dislikes': dislikes,
                'comment_count': comments,
                'thumbnail_link': f'https://img.youtube.com/vi/video_{i:08d}/default.jpg',
                'channel_subscribers': channel_subscribers,
                'trending_date': trending_date.strftime('%Y-%m-%d'),
                'publish_date': publish_date.strftime('%Y-%m-%d'),
                'description': f'This is a trending video about {np.random.choice(categories).lower()}.'
            })
        
        return pd.DataFrame(data)
    
    def _process_youtube_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process YouTube dataset with engagement metrics"""
        
        # Calculate engagement metrics
        df = self.calculate_engagement_metrics(df)
        
        # Add temporal features
        df = self._add_temporal_features(df)
        
        # Add content features
        df = self._add_content_features(df)
        
        # Add channel features
        df = self._add_channel_features(df)
        
        return df
    
    def calculate_engagement_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive engagement metrics for YouTube data
        
        Args:
            df (pd.DataFrame): DataFrame with YouTube data
            
        Returns:
            pd.DataFrame: DataFrame with added engagement metrics
        """
        # Basic engagement rates
        df['like_rate'] = df['likes'] / (df['views'] + 1)
        df['dislike_rate'] = df['dislikes'] / (df['views'] + 1)
        df['comment_rate'] = df['comment_count'] / (df['views'] + 1)
        
        # Total engagement
        df['total_engagement'] = df['likes'] + df['dislikes'] + df['comment_count']
        df['engagement_rate'] = df['total_engagement'] / (df['views'] + 1)
        
        # Engagement ratios
        df['like_dislike_ratio'] = df['likes'] / (df['dislikes'] + 1)
        df['comment_like_ratio'] = df['comment_count'] / (df['likes'] + 1)
        
        # Quality indicators
        df['like_to_view_ratio'] = df['likes'] / (df['views'] + 1)
        df['comment_to_view_ratio'] = df['comment_count'] / (df['views'] + 1)
        
        # Virality indicators
        df['virality_score'] = df['likes'] * df['comment_count'] / (df['views'] + 1)
        df['engagement_intensity'] = df['total_engagement'] / np.log1p(df['views'])
        
        # Sentiment proxy (like/dislike ratio)
        df['sentiment_proxy'] = (df['likes'] - df['dislikes']) / (df['likes'] + df['dislikes'] + 1)
        
        # Popularity indicators
        df['popularity_score'] = (
            df['views'] * 0.4 + 
            df['likes'] * 0.3 + 
            df['comment_count'] * 0.2 + 
            df['channel_subscribers'] * 0.1
        )
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to the dataset"""
        
        # Convert dates
        if 'trending_date' in df.columns:
            df['trending_date'] = pd.to_datetime(df['trending_date'])
            df['trending_day_of_week'] = df['trending_date'].dt.dayofweek
            df['trending_month'] = df['trending_date'].dt.month
            df['trending_year'] = df['trending_date'].dt.year
            df['is_weekend'] = (df['trending_date'].dt.dayofweek >= 5).astype(int)
        
        if 'publish_date' in df.columns:
            df['publish_date'] = pd.to_datetime(df['publish_date'])
            df['publish_day_of_week'] = df['publish_date'].dt.dayofweek
            df['publish_month'] = df['publish_date'].dt.month
            df['publish_year'] = df['publish_date'].dt.year
        
        # Time to trend (if both dates available)
        if 'trending_date' in df.columns and 'publish_date' in df.columns:
            df['days_to_trend'] = (df['trending_date'] - df['publish_date']).dt.days
            df['hours_to_trend'] = df['days_to_trend'] * 24
        
        return df
    
    def _add_content_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add content-based features"""
        
        # Title features
        if 'title' in df.columns:
            df['title_length'] = df['title'].str.len()
            df['title_word_count'] = df['title'].str.split().str.len()
            df['title_has_numbers'] = df['title'].str.contains(r'\d').astype(int)
            df['title_uppercase_ratio'] = df['title'].str.count(r'[A-Z]') / df['title'].str.len()
        
        # Tag features
        if 'tags' in df.columns:
            df['tag_count'] = df['tags'].str.split('|').str.len()
            df['has_tags'] = (df['tag_count'] > 0).astype(int)
        
        # Description features
        if 'description' in df.columns:
            df['description_length'] = df['description'].str.len()
            df['description_word_count'] = df['description'].str.split().str.len()
            df['has_description'] = (df['description_length'] > 0).astype(int)
        
        return df
    
    def _add_channel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add channel-based features"""
        
        if 'channel_subscribers' in df.columns:
            # Channel influence
            df['channel_influence'] = np.log1p(df['channel_subscribers'])
            
            # Subscriber to view ratio
            df['subscriber_view_ratio'] = df['channel_subscribers'] / (df['views'] + 1)
            
            # Channel size categories
            df['channel_size'] = pd.cut(
                df['channel_subscribers'],
                bins=[0, 1000, 10000, 100000, 1000000, float('inf')],
                labels=['tiny', 'small', 'medium', 'large', 'mega']
            )
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = None,
                       feature_types: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for machine learning models
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Target variable column name
            feature_types (List[str]): Types of features to include
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Processed features and feature list
        """
        if feature_types is None:
            feature_types = ['numerical', 'categorical', 'temporal', 'text']
        
        processed_df = df.copy()
        feature_columns = []
        
        # Numerical features
        if 'numerical' in feature_types:
            numerical_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column:
                numerical_cols = [col for col in numerical_cols if col != target_column]
            
            # Handle missing values
            for col in numerical_cols:
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
            
            feature_columns.extend(numerical_cols)
        
        # Categorical features
        if 'categorical' in feature_types:
            categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns.tolist()
            if target_column:
                categorical_cols = [col for col in categorical_cols if col != target_column]
            
            # One-hot encode categorical variables
            for col in categorical_cols:
                if processed_df[col].nunique() < 50:  # Only encode low-cardinality columns
                    dummies = pd.get_dummies(processed_df[col], prefix=col, drop_first=True)
                    processed_df = pd.concat([processed_df, dummies], axis=1)
                    feature_columns.extend(dummies.columns.tolist())
        
        # Temporal features (already processed in previous steps)
        if 'temporal' in feature_types:
            temporal_cols = [col for col in processed_df.columns if 
                           any(keyword in col.lower() for keyword in ['day', 'month', 'year', 'date', 'time'])]
            if target_column:
                temporal_cols = [col for col in temporal_cols if col != target_column]
            
            feature_columns.extend(temporal_cols)
        
        # Text features (already cleaned and processed)
        if 'text' in feature_types:
            text_cols = [col for col in processed_df.columns if col in ['cleaned_text', 'text_length', 'word_count']]
            if target_column:
                text_cols = [col for col in text_cols if col != target_column]
            
            feature_columns.extend(text_cols)
        
        # Remove duplicates and ensure all features exist
        feature_columns = list(set(feature_columns))
        feature_columns = [col for col in feature_columns if col in processed_df.columns]
        
        # Return only the features and target
        if target_column and target_column in processed_df.columns:
            result_df = processed_df[feature_columns + [target_column]]
        else:
            result_df = processed_df[feature_columns]
        
        return result_df, feature_columns
    
    def save_processed_data(self, df: pd.DataFrame, file_path: str,
                          metadata: Dict[str, Any] = None) -> None:
        """
        Save processed dataset with metadata
        
        Args:
            df (pd.DataFrame): Processed dataset
            file_path (str): Path to save the dataset
            metadata (Dict[str, Any]): Metadata to save
        """
        # Save main dataset
        df.to_csv(file_path, index=False)
        
        # Save metadata if provided
        if metadata:
            metadata_path = file_path.replace('.csv', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        print(f"Saved processed dataset to {file_path}")
        if metadata:
            print(f"Saved metadata to {metadata_path}")
    
    def load_processed_data(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load processed dataset with metadata
        
        Args:
            file_path (str): Path to the processed dataset
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Dataset and metadata
        """
        # Load main dataset
        df = pd.read_csv(file_path)
        
        # Load metadata if available
        metadata_path = file_path.replace('.csv', '_metadata.json')
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return df, metadata


# Example usage and demonstration
def main():
    """
    Example usage of the DatasetPreprocessor
    """
    print("=== Dataset Preprocessing Demo ===\n")
    
    # Initialize preprocessor
    preprocessor = DatasetPreprocessor()
    
    # Process IMDB dataset
    print("1. Processing IMDB dataset...")
    imdb_df, imdb_metadata = preprocessor.load_imdb_dataset(
        sample_size=1000, 
        download_sample=True
    )
    
    print(f"IMDB dataset shape: {imdb_df.shape}")
    print(f"Columns: {list(imdb_df.columns)}")
    print(f"Sample review: {imdb_df['text'].iloc[0][:100]}...")
    print(f"Cleaned text: {imdb_df['cleaned_text'].iloc[0][:100]}...")
    print()
    
    # Process YouTube dataset
    print("2. Processing YouTube dataset...")
    youtube_df, youtube_metadata = preprocessor.load_youtube_dataset(
        sample_size=2000,
        download_sample=True
    )
    
    print(f"YouTube dataset shape: {youtube_df.shape}")
    print(f"Columns: {list(youtube_df.columns)}")
    print(f"Sample engagement metrics:")
    sample_row = youtube_df.iloc[0]
    print(f"  Views: {sample_row['views']:,}")
    print(f"  Likes: {sample_row['likes']:,}")
    print(f"  Engagement Rate: {sample_row['engagement_rate']:.4f}")
    print()
    
    # Prepare features for ML
    print("3. Preparing features for machine learning...")
    
    # For sentiment analysis
    imdb_features, imdb_feature_list = preprocessor.prepare_features(
        imdb_df, 
        target_column='sentiment_label',
        feature_types=['numerical', 'text']
    )
    print(f"IMDB features shape: {imdb_features.shape}")
    print(f"IMDB feature count: {len(imdb_feature_list)}")
    
    # For popularity prediction
    youtube_features, youtube_feature_list = preprocessor.prepare_features(
        youtube_df,
        target_column='views',
        feature_types=['numerical', 'categorical', 'temporal']
    )
    print(f"YouTube features shape: {youtube_features.shape}")
    print(f"YouTube feature count: {len(youtube_feature_list)}")
    
    # Save processed data
    print("\n4. Saving processed data...")
    preprocessor.save_processed_data(imdb_df, 'processed_imdb_dataset.csv', imdb_metadata)
    preprocessor.save_processed_data(youtube_df, 'processed_youtube_dataset.csv', youtube_metadata)
    
    print("\n=== Demo completed successfully! ===")


if __name__ == "__main__":
    main()
