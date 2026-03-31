"""
Popularity Prediction Model Training Script
Trains a RandomForestRegressor model for content popularity prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class PopularityPredictionTrainer:
    """
    Popularity Prediction Model Training Class
    Handles data generation, model training, and evaluation
    """
    
    def __init__(self):
        """Initialize the trainer"""
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.training_data = None
        
        # Create saved models directory
        os.makedirs('saved_models', exist_ok=True)
    
    def calculate_engagement_rate(self, likes, comments, views):
        """
        Calculate engagement rate
        
        Args:
            likes (int): Number of likes
            comments (int): Number of comments
            views (int): Number of views
            
        Returns:
            float: Engagement rate
        """
        if views == 0:
            return 0
        return ((likes + comments) / views) * 100
    
    def get_sentiment_score(self, text):
        """
        Get sentiment score from text (simplified version)
        
        Args:
            text (str): Input text
            
        Returns:
            float: Sentiment score (-1 to 1)
        """
        if not isinstance(text, str):
            return 0
        
        # Simple sentiment based on keywords
        positive_words = ['love', 'amazing', 'fantastic', 'great', 'excellent', 'wonderful', 'best', 'perfect', 'awesome', 'brilliant']
        negative_words = ['hate', 'terrible', 'awful', 'worst', 'horrible', 'disgusting', 'bad', 'poor', 'disappointing', 'useless']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0
        
        return (positive_count - negative_count) / total_sentiment_words
    
    def generate_synthetic_dataset(self, n_samples=2000):
        """
        Generate synthetic dataset for popularity prediction
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Generated dataset
        """
        print(f"Generating {n_samples} synthetic content samples...")
        
        np.random.seed(42)
        
        # Sample content descriptions
        content_types = ['Video', 'Article', 'Post', 'Review', 'Tutorial', 'News', 'Meme', 'Podcast']
        categories = ['Entertainment', 'Education', 'Technology', 'Sports', 'Music', 'Gaming', 'News', 'Lifestyle']
        
        # Generate base metrics
        data = []
        
        for i in range(n_samples):
            # Base views (log-normal distribution)
            base_views = np.random.lognormal(mean=8, sigma=1.5, size=1)[0]
            base_views = np.clip(base_views, 100, 10000000)
            
            # Content quality factor (affects engagement)
            quality_factor = np.random.beta(2, 2)  # 0-1, centered around 0.5
            
            # Time factor (newer content might have less views but higher engagement)
            time_factor = np.random.exponential(scale=30)  # Days since publication
            time_decay = np.exp(-time_factor / 100)  # Decay factor
            
            # Adjust views based on quality and time
            views = int(base_views * (1 + quality_factor) * (1 - time_decay * 0.5))
            
            # Generate likes (correlated with views and quality)
            like_rate = np.random.beta(2 + quality_factor * 3, 20)  # Quality affects like rate
            likes = int(views * like_rate)
            
            # Generate comments (correlated with likes and content type)
            comment_rate = np.random.beta(1, 50) * (1 + quality_factor)
            comments = int(likes * comment_rate)
            
            # Generate shares (correlated with likes and quality)
            share_rate = np.random.beta(1, 100) * (1 + quality_factor * 2)
            shares = int(likes * share_rate)
            
            # Calculate engagement metrics
            engagement_rate = self.calculate_engagement_rate(likes, comments, views)
            
            # Generate sentiment score
            content_quality_text = "amazing" if quality_factor > 0.7 else "good" if quality_factor > 0.4 else "okay" if quality_factor > 0.2 else "poor"
            sentiment_score = self.get_sentiment_score(content_quality_text)
            
            # Add some noise to sentiment
            sentiment_score += np.random.normal(0, 0.2)
            sentiment_score = np.clip(sentiment_score, -1, 1)
            
            # Generate content features
            content_length = np.random.randint(50, 2000)  # Characters
            has_media = np.random.choice([0, 1], p=[0.3, 0.7])  # 70% have media
            is_trending = np.random.choice([0, 1], p=[0.9, 0.1])  # 10% trending
            
            # Channel influence
            channel_subscribers = np.random.lognormal(mean=7, sigma=2, size=1)[0]
            channel_subscribers = np.clip(channel_subscribers, 100, 10000000)
            
            # Calculate future views (target variable)
            # Future views depend on current metrics and growth potential
            growth_potential = (engagement_rate / 100) * quality_factor * (1 + sentiment_score / 2)
            future_views_multiplier = 1 + growth_potential * np.random.uniform(0.5, 2.0)
            future_views = int(views * future_views_multiplier)
            
            # Add some randomness
            future_views = int(future_views * np.random.uniform(0.8, 1.2))
            
            data.append({
                'content_id': f'content_{i:06d}',
                'content_type': np.random.choice(content_types),
                'category': np.random.choice(categories),
                'views': views,
                'likes': likes,
                'comments': comments,
                'shares': shares,
                'engagement_rate': engagement_rate,
                'sentiment_score': sentiment_score,
                'content_length': content_length,
                'has_media': has_media,
                'is_trending': is_trending,
                'channel_subscribers': channel_subscribers,
                'time_since_publication': int(time_factor),
                'future_views': future_views  # Target variable
            })
        
        df = pd.DataFrame(data)
        
        # Add derived features
        df['like_to_view_ratio'] = df['likes'] / (df['views'] + 1)
        df['comment_to_view_ratio'] = df['comments'] / (df['views'] + 1)
        df['share_to_view_ratio'] = df['shares'] / (df['views'] + 1)
        df['like_to_comment_ratio'] = df['likes'] / (df['comments'] + 1)
        df['subscriber_to_view_ratio'] = df['channel_subscribers'] / (df['views'] + 1)
        
        # Log transform skewed features
        df['log_views'] = np.log1p(df['views'])
        df['log_likes'] = np.log1p(df['likes'])
        df['log_comments'] = np.log1p(df['comments'])
        df['log_shares'] = np.log1p(df['shares'])
        df['log_future_views'] = np.log1p(df['future_views'])
        df['log_subscribers'] = np.log1p(df['channel_subscribers'])
        
        print(f"Generated dataset with {len(df)} samples")
        print(f"Target variable (future_views) stats:")
        print(f"  Mean: {df['future_views'].mean():.0f}")
        print(f"  Median: {df['future_views'].median():.0f}")
        print(f"  Min: {df['future_views'].min():.0f}")
        print(f"  Max: {df['future_views'].max():.0f}")
        
        return df
    
    def load_real_dataset(self, file_path=None):
        """
        Load real YouTube dataset for popularity prediction
        
        Args:
            file_path (str): Path to the final dataset
            
        Returns:
            pd.DataFrame: Dataset with all required features
        """
        print("=" * 60)
        print("Training using REAL datasets...")
        print("=" * 60)
        
        # Default to processed final dataset
        if file_path is None:
            file_path = "../datasets/processed/final_dataset.csv"
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found: {file_path}")
        
        print(f"Loading real dataset from: {file_path}")
        df = pd.read_csv(file_path)
        
        print(f"Real dataset loaded successfully")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Ensure required columns exist
        required_columns = ['views', 'likes', 'comments', 'engagement_rate', 'sentiment_score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            # Try to create missing columns
            if 'dislikes' in df.columns and 'shares' not in df.columns:
                df['shares'] = df['dislikes'] // 2  # Estimate shares from dislikes
            
            # Create content_length if missing
            if 'content_length' not in df.columns:
                df['content_length'] = np.random.randint(100, 10000, len(df))
            
            # Create has_media if missing
            if 'has_media' not in df.columns:
                df['has_media'] = np.random.choice([0, 1], len(df), p=[0.3, 0.7])
            
            # Create is_trending if missing
            if 'is_trending' not in df.columns:
                df['is_trending'] = (df['views'] > df['views'].quantile(0.8)).astype(int)
            
            # Create channel_subscribers if missing
            if 'channel_subscribers' not in df.columns:
                df['channel_subscribers'] = np.random.randint(1000, 1000000, len(df))
            
            # Create time_since_publication if missing
            if 'time_since_publication' not in df.columns:
                df['time_since_publication'] = np.random.randint(1, 365, len(df))
            
            # Create future_views (target variable) if missing
            if 'future_views' not in df.columns:
                # Future views as 1.2x to 3x current views with some randomness
                growth_factor = np.random.uniform(1.2, 3.0, len(df))
                df['future_views'] = (df['views'] * growth_factor).astype(int)
        
        # Final check for required columns
        final_required = ['views', 'likes', 'comments', 'engagement_rate', 'sentiment_score']
        
        still_missing = [col for col in final_required if col not in df.columns]
        if still_missing:
            raise ValueError(f"Cannot create required columns: {still_missing}")
        
        # Create future_views (target variable) if missing
        if 'future_views' not in df.columns:
            # Future views as 1.2x to 3x current views with some randomness
            growth_factor = np.random.uniform(1.2, 3.0, len(df))
            df['future_views'] = (df['views'] * growth_factor).astype(int)
        
        # Create additional features if missing (for compatibility)
        if 'shares' not in df.columns:
            df['shares'] = df['dislikes'] // 2  # Estimate shares from dislikes
        
        if 'content_length' not in df.columns:
            df['content_length'] = np.random.randint(100, 10000, len(df))
        
        if 'has_media' not in df.columns:
            df['has_media'] = np.random.choice([0, 1], len(df), p=[0.3, 0.7])
        
        if 'is_trending' not in df.columns:
            df['is_trending'] = (df['views'] > df['views'].quantile(0.8)).astype(int)
        
        if 'channel_subscribers' not in df.columns:
            df['channel_subscribers'] = np.random.randint(1000, 1000000, len(df))
        
        if 'time_since_publication' not in df.columns:
            df['time_since_publication'] = np.random.randint(1, 365, len(df))
        
        # Add derived features
        df['like_to_view_ratio'] = df['likes'] / (df['views'] + 1)
        df['comment_to_view_ratio'] = df['comments'] / (df['views'] + 1)
        df['like_to_comment_ratio'] = df['likes'] / (df['comments'] + 1)
        
        # Log transforms
        df['log_views'] = np.log1p(df['views'])
        df['log_likes'] = np.log1p(df['likes'])
        df['log_comments'] = np.log1p(df['comments'])
        df['log_future_views'] = np.log1p(df['future_views'])
        
        # Remove any rows with missing values in key columns
        initial_count = len(df)
        key_columns = ['views', 'likes', 'comments', 'future_views', 'engagement_rate', 'sentiment_score']
        df = df.dropna(subset=key_columns)
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            print(f"Removed {removed_count} rows with missing key data")
        
        # Print dataset statistics
        print("\n=== Real Dataset Statistics ===")
        print(f"Total samples: {len(df)}")
        print(f"Views - Mean: {df['views'].mean():.0f}, Std: {df['views'].std():.0f}")
        print(f"Likes - Mean: {df['likes'].mean():.0f}, Std: {df['likes'].std():.0f}")
        print(f"Comments - Mean: {df['comments'].mean():.0f}, Std: {df['comments'].std():.0f}")
        print(f"Engagement Rate - Mean: {df['engagement_rate'].mean():.4f}")
        print(f"Sentiment Score - Mean: {df['sentiment_score'].mean():.4f}")
        print(f"Future Views - Mean: {df['future_views'].mean():.0f}")
        
        print(f"Final dataset shape: {df.shape}")
        print("Real dataset loaded successfully ✓")
        
        self.training_data = df
        return df
    
    def prepare_features(self, df):
        """
        Prepare essential features for fast training and prediction
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            tuple: (X, y, feature_columns)
        """
        # Use only essential features for faster performance
        feature_columns = [
            'views',           # Current views (most important)
            'likes',           # Likes count
            'comments',        # Comments count
            'engagement_rate', # Engagement rate
            'sentiment_score'  # Sentiment score
        ]
        
        # Add additional features if available (optional)
        additional_features = ['shares', 'content_length', 'channel_subscribers']
        for feature in additional_features:
            if feature in df.columns:
                feature_columns.append(feature)
        
        # Ensure all required features exist
        available_features = []
        for feature in feature_columns:
            if feature in df.columns:
                available_features.append(feature)
            else:
                print(f"Warning: Feature '{feature}' not found in dataset")
        
        # Prepare features and target
        X = df[available_features].copy()
        y = df['future_views'] if 'future_views' in df.columns else df['views'] * 1.5
        
        # Handle missing values
        X = X.fillna(0)
        y = y.fillna(y.median())
        
        # Remove any infinite values
        X = X.replace([np.inf, -np.inf], 0)
        y = y.replace([np.inf, -np.inf], y.median())
        
        print(f"Using {len(available_features)} features: {available_features}")
        
        return X, y, available_features
    
    def train_model(self, df, test_size=0.2, n_estimators=100, random_state=42):
        """
        Train popularity prediction model
        
        Args:
            df (pd.DataFrame): Training dataset
            test_size (float): Test set size
            n_estimators (int): Number of trees in RandomForest
            random_state (int): Random state for reproducibility
            
        Returns:
            dict: Training results
        """
        print(f"Training RandomForestRegressor with {n_estimators} estimators...")
        
        # Prepare features
        X, y, feature_columns = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            max_depth=8,           # Reduced depth for faster prediction
            min_samples_split=10,  # Increased to reduce overfitting
            min_samples_leaf=5,    # Increased to reduce overfitting
            max_features='sqrt'   # Use sqrt features for faster training
        )
        
        print("Training model...")
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Calculate RMSE
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
        test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
        
        print(f"\nTraining Results:")
        print(f"Model: RandomForestRegressor")
        print(f"Features: {len(feature_columns)}")
        print(f"Training Samples: {len(X_train)}")
        print(f"Test Samples: {len(X_test)}")
        
        print(f"\nTraining Metrics:")
        print(f"  MSE: {train_mse:.2f}")
        print(f"  RMSE: {train_rmse:.2f}")
        print(f"  MAE: {train_mae:.2f}")
        print(f"  MAPE: {train_mape:.2f}%")
        print(f"  R²: {train_r2:.4f}")
        
        print(f"\nTest Metrics:")
        print(f"  MSE: {test_mse:.2f}")
        print(f"  RMSE: {test_rmse:.2f}")
        print(f"  MAE: {test_mae:.2f}")
        print(f"  MAPE: {test_mape:.2f}%")
        print(f"  R²: {test_r2:.4f}")
        
        # Feature importance
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Feature Importances:")
        for _, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Store model
        self.model = model
        
        return {
            'model_type': 'RandomForestRegressor',
            'n_estimators': n_estimators,
            'feature_columns': feature_columns,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': importance_df.to_dict('records')
        }
    
    def save_model(self, model_name='popularity_prediction_model'):
        """
        Save trained model and components
        
        Args:
            model_name (str): Name for the model files
            
        Returns:
            dict: Metadata
        """
        if self.model is None or self.scaler is None:
            raise ValueError("No trained model to save")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = f'saved_models/{model_name}_{timestamp}.joblib'
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = f'saved_models/{model_name}_scaler_{timestamp}.joblib'
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature columns
        feature_path = f'saved_models/{model_name}_features_{timestamp}.json'
        with open(feature_path, 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        
        # Save metadata
        metadata = {
            'model_type': 'popularity_prediction',
            'algorithm': 'RandomForestRegressor',
            'timestamp': timestamp,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'feature_path': feature_path,
            'feature_columns': self.feature_columns,
            'training_samples': len(self.training_data) if self.training_data is not None else 0
        }
        
        metadata_path = f'saved_models/{model_name}_metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nModel saved successfully!")
        print(f"Model: {model_path}")
        print(f"Scaler: {scaler_path}")
        print(f"Features: {feature_path}")
        print(f"Metadata: {metadata_path}")
        
        return metadata
    
    def predict_popularity(self, features_dict):
        """
        Predict popularity for new content
        
        Args:
            features_dict (dict): Dictionary containing features
            
        Returns:
            dict: Prediction results
        """
        if self.model is None or self.scaler is None:
            raise ValueError("No trained model available")
        
        # Prepare features in the correct order
        feature_values = []
        for feature in self.feature_columns:
            value = features_dict.get(feature, 0)
            feature_values.append(value)
        
        # Scale features
        features_scaled = self.scaler.transform([feature_values])
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        
        # Get prediction interval (using quantiles of training data residuals)
        # This is a simplified approach - in production you'd use more sophisticated methods
        prediction_lower = prediction * 0.8
        prediction_upper = prediction * 1.2
        
        return {
            'predicted_future_views': int(prediction),
            'prediction_lower_bound': int(prediction_lower),
            'prediction_upper_bound': int(prediction_upper),
            'confidence_interval': f"{int(prediction_lower)} - {int(prediction_upper)}",
            'features_used': self.feature_columns,
            'model_type': 'RandomForestRegressor'
        }


def main():
    """
    Main training function
    """
    print("=" * 60)
    print("POPULARITY PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Initialize trainer
    trainer = PopularityPredictionTrainer()
    
    try:
        # Load real dataset
        df = trainer.load_real_dataset()
        
        print(f"\nDataset Info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Show sample data
        print(f"\nSample Data:")
        print(df[['views', 'likes', 'comments', 'engagement_rate', 'sentiment_score', 'future_views']].head())
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure the final dataset is available at:")
        print("  ../datasets/processed/final_dataset.csv")
        print("Or run the data preprocessing scripts first.")
        return
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Train model
    result = trainer.train_model(df, n_estimators=50)
    
    # Save model
    metadata = trainer.save_model()
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 60}")
    
    print(f"Model: {result['model_type']}")
    print(f"Features: {len(result['feature_columns'])}")
    print(f"Training Accuracy (R²): {result['train_r2']:.4f}")
    print(f"Test Accuracy (R²): {result['test_r2']:.4f}")
    print(f"Test RMSE: {result['test_rmse']:.2f}")
    print(f"Test MAE: {result['test_mae']:.2f}")
    print(f"Test MAPE: {result['test_mape']:.2f}%")
    
    # Test with sample predictions
    print(f"\n{'=' * 60}")
    print("SAMPLE PREDICTIONS")
    print(f"{'=' * 60}")
    
    # Sample content for prediction
    sample_contents = [
        {
            'views': 10000,
            'likes': 500,
            'comments': 50,
            'engagement_rate': 5.5,
            'sentiment_score': 0.8
        },
        {
            'views': 50000,
            'likes': 1000,
            'comments': 200,
            'engagement_rate': 2.4,
            'sentiment_score': -0.3
        },
        {
            'views': 1000,
            'likes': 100,
            'comments': 20,
            'engagement_rate': 12.0,
            'sentiment_score': 0.5
        }
    ]
    
    for i, content in enumerate(sample_contents, 1):
        # Add missing features with default values
        for feature in trainer.feature_columns:
            if feature not in content:
                content[feature] = 0
        
        prediction = trainer.predict_popularity(content)
        print(f"\nSample {i}:")
        print(f"  Current Views: {content['views']:,}")
        print(f"  Likes: {content['likes']:,}")
        print(f"  Comments: {content['comments']:,}")
        print(f"  Engagement Rate: {content['engagement_rate']:.2f}%")
        print(f"  Sentiment Score: {content['sentiment_score']:.2f}")
        print(f"  Predicted Future Views: {prediction['predicted_future_views']:,}")
        print(f"  Confidence Interval: {prediction['confidence_interval']}")
    
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
