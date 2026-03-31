"""
Data Preprocessing Utilities
Common preprocessing functions for all models in the Hybrid AI System
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Data Preprocessing Utility Class
    Provides common preprocessing functions for text, temporal, and numerical data
    """
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
            print("Warning: NLTK stopwords not available. Run nltk.download('stopwords')")
    
    def clean_text(self, text: str, remove_stopwords: bool = True, 
                   lemmatize: bool = True, min_length: int = 3) -> str:
        """
        Clean and preprocess text data
        
        Args:
            text (str): Input text to clean
            remove_stopwords (bool): Whether to remove stopwords
            lemmatize (bool): Whether to lemmatize words
            min_length (int): Minimum word length to keep
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (keep the text)
        text = re.sub(r'@\w+|#\w+', '', text)
        
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
    
    def extract_text_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract linguistic features from texts
        
        Args:
            texts (List[str]): List of text samples
            
        Returns:
            pd.DataFrame: DataFrame with text features
        """
        features = []
        
        for text in texts:
            if pd.isna(text) or not isinstance(text, str):
                text = ""
            
            # Basic text statistics
            char_count = len(text)
            word_count = len(text.split())
            
            try:
                sentence_count = len(sent_tokenize(text))
            except:
                sentence_count = 1 if text else 0
            
            # Average word length
            words = text.split()
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            # Punctuation count
            punctuation_count = len(re.findall(r'[^\w\s]', text))
            
            # Uppercase count
            uppercase_count = sum(1 for char in text if char.isupper())
            
            # Digit count
            digit_count = sum(1 for char in text if char.isdigit())
            
            # Emoji count
            emoji_count = self._count_emojis(text)
            
            # Readability scores (simplified)
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            features.append({
                'char_count': char_count,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_word_length': avg_word_length,
                'punctuation_count': punctuation_count,
                'uppercase_count': uppercase_count,
                'digit_count': digit_count,
                'emoji_count': emoji_count,
                'avg_sentence_length': avg_sentence_length
            })
        
        return pd.DataFrame(features)
    
    def preprocess_temporal_data(self, df: pd.DataFrame, 
                               timestamp_column: str = 'timestamp') -> pd.DataFrame:
        """
        Preprocess temporal data and extract time-based features
        
        Args:
            df (pd.DataFrame): DataFrame with temporal data
            timestamp_column (str): Name of timestamp column
            
        Returns:
            pd.DataFrame: DataFrame with temporal features
        """
        if timestamp_column not in df.columns:
            return df
        
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Convert to datetime
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
        
        # Extract time features
        df['hour'] = df[timestamp_column].dt.hour
        df['day_of_week'] = df[timestamp_column].dt.dayofweek
        df['day_of_month'] = df[timestamp_column].dt.day
        df['month'] = df[timestamp_column].dt.month
        df['year'] = df[timestamp_column].dt.year
        df['quarter'] = df[timestamp_column].dt.quarter
        
        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
        
        # Business hours
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df (pd.DataFrame): Input DataFrame
            strategy (str): Strategy for handling missing values
                ('mean', 'median', 'mode', 'drop', 'forward_fill', 'backward_fill')
                
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        df = df.copy()
        
        if strategy == 'drop':
            return df.dropna()
        
        for column in df.columns:
            if df[column].isna().any():
                if df[column].dtype in ['int64', 'float64']:
                    if strategy == 'mean':
                        df[column].fillna(df[column].mean(), inplace=True)
                    elif strategy == 'median':
                        df[column].fillna(df[column].median(), inplace=True)
                else:
                    if strategy == 'mode':
                        mode_value = df[column].mode()
                        if not mode_value.empty:
                            df[column].fillna(mode_value[0], inplace=True)
                    elif strategy == 'forward_fill':
                        df[column].fillna(method='ffill', inplace=True)
                    elif strategy == 'backward_fill':
                        df[column].fillna(method='bfill', inplace=True)
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame, 
                       columns: Optional[List[str]] = None,
                       method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers from the dataset
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): Columns to check for outliers
            method (str): Method for outlier detection ('iqr', 'zscore')
            
        Returns:
            pd.DataFrame: DataFrame with outliers removed
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in columns:
            if column not in df.columns:
                continue
            
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            
            elif method == 'zscore':
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                df = df[z_scores < 3]
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, 
                         columns: Optional[List[str]] = None,
                         method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize numerical features
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): Columns to normalize
            method (str): Normalization method ('minmax', 'zscore')
            
        Returns:
            pd.DataFrame: DataFrame with normalized features
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in columns:
            if column not in df.columns:
                continue
            
            if method == 'minmax':
                min_val = df[column].min()
                max_val = df[column].max()
                if max_val != min_val:
                    df[column] = (df[column] - min_val) / (max_val - min_val)
            
            elif method == 'zscore':
                mean_val = df[column].mean()
                std_val = df[column].std()
                if std_val != 0:
                    df[column] = (df[column] - mean_val) / std_val
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                  columns: Optional[List[str]] = None,
                                  method: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): Columns to encode
            method (str): Encoding method ('onehot', 'label')
            
        Returns:
            pd.DataFrame: DataFrame with encoded features
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for column in columns:
            if column not in df.columns:
                continue
            
            if method == 'onehot':
                dummies = pd.get_dummies(df[column], prefix=column)
                df = pd.concat([df, dummies], axis=1)
                df.drop(column, axis=1, inplace=True)
            
            elif method == 'label':
                df[column] = pd.Categorical(df[column]).codes
        
        return df
    
    def _count_emojis(self, text: str) -> int:
        """Count emojis in text"""
        if not text:
            return 0
        
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            "]+", flags=re.UNICODE
        )
        
        return len(emoji_pattern.findall(text))
    
    def create_lag_features(self, df: pd.DataFrame, 
                          columns: List[str], 
                          lags: List[int]) -> pd.DataFrame:
        """
        Create lag features for time series data
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): Columns to create lags for
            lags (List[int]): Lag periods
            
        Returns:
            pd.DataFrame: DataFrame with lag features
        """
        df = df.copy()
        
        for column in columns:
            if column not in df.columns:
                continue
            
            for lag in lags:
                df[f'{column}_lag_{lag}'] = df[column].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, 
                              columns: List[str], 
                              windows: List[int]) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): Columns to create rolling features for
            windows (List[int]): Window sizes
            
        Returns:
            pd.DataFrame: DataFrame with rolling features
        """
        df = df.copy()
        
        for column in columns:
            if column not in df.columns:
                continue
            
            for window in windows:
                df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window).mean()
                df[f'{column}_rolling_std_{window}'] = df[column].rolling(window).std()
                df[f'{column}_rolling_min_{window}'] = df[column].rolling(window).min()
                df[f'{column}_rolling_max_{window}'] = df[column].rolling(window).max()
        
        return df

# Utility functions for common preprocessing tasks
def clean_dataset(df: pd.DataFrame, 
                 text_columns: Optional[List[str]] = None,
                 timestamp_column: Optional[str] = None,
                 handle_missing: bool = True,
                 remove_outliers: bool = True) -> pd.DataFrame:
    """
    Comprehensive dataset cleaning function
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_columns (List[str]): Text columns to clean
        timestamp_column (str): Timestamp column for temporal features
        handle_missing (bool): Whether to handle missing values
        remove_outliers (bool): Whether to remove outliers
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    preprocessor = DataPreprocessor()
    
    # Clean text columns
    if text_columns:
        for column in text_columns:
            if column in df.columns:
                df[column] = df[column].apply(preprocessor.clean_text)
    
    # Process temporal data
    if timestamp_column and timestamp_column in df.columns:
        df = preprocessor.preprocess_temporal_data(df, timestamp_column)
    
    # Handle missing values
    if handle_missing:
        df = preprocessor.handle_missing_values(df)
    
    # Remove outliers
    if remove_outliers:
        df = preprocessor.remove_outliers(df)
    
    return df

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and return statistics
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        Dict[str, Any]: Data quality report
    """
    report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
    }
    
    # Add statistics for numeric columns
    numeric_stats = {}
    for column in report['numeric_columns']:
        numeric_stats[column] = {
            'mean': df[column].mean(),
            'std': df[column].std(),
            'min': df[column].min(),
            'max': df[column].max(),
            'zeros': (df[column] == 0).sum()
        }
    
    report['numeric_statistics'] = numeric_stats
    
    return report
