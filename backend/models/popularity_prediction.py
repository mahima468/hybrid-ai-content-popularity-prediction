"""
Content Popularity Prediction Model
Predicts content popularity using sentiment analysis and engagement metrics
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
from datetime import datetime, timedelta

class PopularityPredictionModel:
    """
    Content Popularity Prediction Model Class
    Predicts content popularity based on sentiment, engagement, and content features
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the popularity prediction model
        
        Args:
            model_type (str): Type of model to use
                ('random_forest', 'gradient_boosting', 'linear', 'ridge')
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_regression, k=20)
        self.model = None
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        else:
            raise ValueError("Model type must be 'random_forest', 'gradient_boosting', 'linear', or 'ridge'")
    
    def extract_popularity_features(self, content_data, sentiment_data=None, engagement_data=None):
        """
        Extract features for popularity prediction
        
        Args:
            content_data (pd.DataFrame): Content metadata
            sentiment_data (pd.DataFrame): Sentiment analysis results
            engagement_data (pd.DataFrame): Engagement metrics
            
        Returns:
            pd.DataFrame: Features for prediction
        """
        features = content_data.copy()
        
        # Content-based features
        if 'content_length' in features.columns:
            features['content_length_log'] = np.log1p(features['content_length'])
        
        if 'has_media' in features.columns:
            features['has_media'] = features['has_media'].astype(int)
        
        # Time-based features
        if 'post_time' in features.columns:
            features['post_time'] = pd.to_datetime(features['post_time'])
            features['hour_of_day'] = features['post_time'].dt.hour
            features['day_of_week'] = features['post_time'].dt.dayofweek
            features['is_weekend'] = (features['post_time'].dt.dayofweek >= 5).astype(int)
        
        # Sentiment features
        if sentiment_data is not None:
            sentiment_agg = sentiment_data.groupby('content_id').agg({
                'sentiment': ['mean', 'std', 'count'],
                'confidence': 'mean'
            }).round(3)
            
            sentiment_agg.columns = [
                'avg_sentiment', 'sentiment_std', 'sentiment_count', 'avg_confidence'
            ]
            
            # Add sentiment distribution features
            if 'sentiment_label' in sentiment_data.columns:
                sentiment_dist = pd.crosstab(
                    sentiment_data['content_id'], 
                    sentiment_data['sentiment_label'], 
                    normalize='index'
                ).round(3)
                
                sentiment_dist.columns = [f'sentiment_{col}' for col in sentiment_dist.columns]
                features = pd.merge(features, sentiment_dist, left_index=True, right_index=True, how='left')
            
            features = pd.merge(features, sentiment_agg, left_index=True, right_index=True, how='left')
        
        # Engagement features
        if engagement_data is not None:
            engagement_agg = engagement_data.groupby('content_id').agg({
                'like_count': ['sum', 'mean', 'std'],
                'comment_count': ['sum', 'mean'],
                'share_count': ['sum', 'mean'],
                'view_count': ['sum', 'mean'],
                'is_fake_engagement': 'sum'
            }).round(3)
            
            engagement_agg.columns = [
                'total_likes', 'avg_likes', 'std_likes',
                'total_comments', 'avg_comments',
                'total_shares', 'avg_shares',
                'total_views', 'avg_views',
                'fake_engagement_count'
            ]
            
            # Calculate engagement ratios
            engagement_agg['like_to_view_ratio'] = (
                engagement_agg['total_likes'] / (engagement_agg['total_views'] + 1)
            )
            engagement_agg['comment_to_view_ratio'] = (
                engagement_agg['total_comments'] / (engagement_agg['total_views'] + 1)
            )
            engagement_agg['share_to_view_ratio'] = (
                engagement_agg['total_shares'] / (engagement_agg['total_views'] + 1)
            )
            engagement_agg['fake_engagement_ratio'] = (
                engagement_agg['fake_engagement_count'] / (
                    engagement_agg['total_likes'] + engagement_agg['total_comments'] + 1
                )
            )
            
            features = pd.merge(features, engagement_agg, left_index=True, right_index=True, how='left')
        
        # User/author features
        if 'author_followers' in features.columns:
            features['author_followers_log'] = np.log1p(features['author_followers'])
            features['author_following_log'] = np.log1p(features.get('author_following', 1))
            features['follower_to_following_ratio'] = (
                features['author_followers'] / (features.get('author_following', 1) + 1)
            )
        
        # Fill missing values
        features = features.fillna(0)
        
        return features
    
    def train(self, content_data, target_variable, sentiment_data=None, engagement_data=None):
        """
        Train the popularity prediction model
        
        Args:
            content_data (pd.DataFrame): Content metadata
            target_variable (str): Name of target variable (e.g., 'future_popularity')
            sentiment_data (pd.DataFrame): Sentiment analysis results
            engagement_data (pd.DataFrame): Engagement metrics
            
        Returns:
            dict: Training results and metrics
        """
        # Extract features
        features = self.extract_popularity_features(content_data, sentiment_data, engagement_data)
        
        # Remove target variable from features
        if target_variable in features.columns:
            X = features.drop(columns=[target_variable])
        else:
            X = features
        
        y = content_data[target_variable]
        
        # Remove non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        
        # Handle missing target values
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Train model
        self.model.fit(X_train_selected, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_selected)
        y_test_pred = self.model.predict(X_test_selected)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_selected, y_train, cv=5, scoring='r2')
        
        self.is_trained = True
        
        # Get feature names after selection
        selected_features = X.columns[self.feature_selector.get_support()]
        
        return {
            'training_metrics': {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_mean_r2': cv_scores.mean(),
                'cv_std_r2': cv_scores.std()
            },
            'feature_importance': self._get_feature_importance(selected_features),
            'selected_features': selected_features.tolist(),
            'predictions': {
                'train_predictions': y_train_pred.tolist(),
                'test_predictions': y_test_pred.tolist(),
                'actual_test': y_test.tolist()
            }
        }
    
    def predict(self, content_data, sentiment_data=None, engagement_data=None):
        """
        Predict popularity for new content
        
        Args:
            content_data (pd.DataFrame): Content metadata
            sentiment_data (pd.DataFrame): Sentiment analysis results
            engagement_data (pd.DataFrame): Engagement metrics
            
        Returns:
            dict: Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        features = self.extract_popularity_features(content_data, sentiment_data, engagement_data)
        
        # Keep only numeric columns
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        features = features[numeric_columns]
        
        # Scale and select features
        features_scaled = self.scaler.transform(features)
        features_selected = self.feature_selector.transform(features_scaled)
        
        # Make predictions
        predictions = self.model.predict(features_selected)
        
        # Create results
        results = pd.DataFrame({
            'content_id': content_data.index if hasattr(content_data, 'index') else range(len(predictions)),
            'predicted_popularity': predictions,
            'prediction_timestamp': datetime.now()
        })
        
        # Add confidence intervals if model supports it
        if hasattr(self.model, 'predict') and hasattr(self.model, 'estimators_'):
            # For ensemble models, calculate prediction intervals
            tree_predictions = np.array([
                tree.predict(features_selected) for tree in self.model.estimators_
            ])
            prediction_std = np.std(tree_predictions, axis=0)
            results['prediction_std'] = prediction_std
            results['lower_bound'] = predictions - 1.96 * prediction_std
            results['upper_bound'] = predictions + 1.96 * prediction_std
        
        return {
            'predictions': results,
            'summary': {
                'mean_predicted_popularity': float(np.mean(predictions)),
                'std_predicted_popularity': float(np.std(predictions)),
                'min_predicted_popularity': float(np.min(predictions)),
                'max_predicted_popularity': float(np.max(predictions))
            }
        }
    
    def _get_feature_importance(self, feature_names):
        """Get feature importance from the model"""
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, self.model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(self.model, 'coef_'):
            importance_dict = dict(zip(feature_names, np.abs(self.model.coef_)))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        return {}
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
