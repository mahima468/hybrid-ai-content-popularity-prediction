"""
Fake Engagement Detection Model
Detects fake engagement patterns using machine learning and statistical analysis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from datetime import datetime, timedelta
import re

class EngagementDetectionModel:
    """
    Fake Engagement Detection Model Class
    Identifies patterns of fake engagement using various ML techniques
    """
    
    def __init__(self, detection_method='isolation_forest'):
        """
        Initialize the engagement detection model
        
        Args:
            detection_method (str): Method to use for detection
                ('isolation_forest', 'random_forest', 'dbscan')
        """
        self.detection_method = detection_method
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        
        # Initialize model based on method
        if detection_method == 'isolation_forest':
            self.model = IsolationForest(contamination=0.1, random_state=42)
        elif detection_method == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif detection_method == 'dbscan':
            self.model = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError("Method must be 'isolation_forest', 'random_forest', or 'dbscan'")
    
    def extract_engagement_features(self, engagement_data):
        """
        Extract features from engagement data for fake engagement detection
        
        Args:
            engagement_data (pd.DataFrame): DataFrame containing engagement data
            
        Returns:
            pd.DataFrame: DataFrame with extracted features
        """
        features = pd.DataFrame()
        
        # Time-based features
        if 'timestamp' in engagement_data.columns:
            engagement_data['timestamp'] = pd.to_datetime(engagement_data['timestamp'])
            features['hour_of_day'] = engagement_data['timestamp'].dt.hour
            features['day_of_week'] = engagement_data['timestamp'].dt.dayofweek
            features['is_weekend'] = (engagement_data['timestamp'].dt.dayofweek >= 5).astype(int)
        
        # User behavior features
        if 'user_id' in engagement_data.columns:
            user_stats = engagement_data.groupby('user_id').agg({
                'user_id': 'count',
                'timestamp': ['min', 'max'] if 'timestamp' in engagement_data.columns else 'count'
            })
            user_stats.columns = ['user_action_count', 'first_action', 'last_action']
            
            # Calculate engagement frequency
            if 'timestamp' in engagement_data.columns:
                user_stats['engagement_timespan_hours'] = (
                    user_stats['last_action'] - user_stats['first_action']
                ).dt.total_seconds() / 3600
                user_stats['actions_per_hour'] = user_stats['user_action_count'] / (
                    user_stats['engagement_timespan_hours'] + 1
                )
            
            # Merge back to main dataframe
            features = pd.merge(features, user_stats, left_index=True, right_index=True, how='left')
        
        # Content interaction patterns
        if 'content_id' in engagement_data.columns:
            content_stats = engagement_data.groupby('content_id').size().reset_index(name='content_engagement_count')
            features = pd.merge(features, content_stats, on='content_id', how='left')
        
        # Text analysis features (if applicable)
        if 'comment_text' in engagement_data.columns:
            features['comment_length'] = engagement_data['comment_text'].str.len()
            features['word_count'] = engagement_data['comment_text'].str.split().str.len()
            features['has_emoji'] = engagement_data['comment_text'].apply(self._has_emoji)
            features['is_repetitive'] = engagement_data['comment_text'].apply(self._is_repetitive)
        
        # Network features
        if 'follower_count' in engagement_data.columns:
            features['follower_to_engagement_ratio'] = (
                engagement_data['follower_count'] / (engagement_data.get('like_count', 1) + 1)
            )
        
        return features.fillna(0)
    
    def detect_anomalies(self, engagement_data):
        """
        Detect fake engagement patterns in the data
        
        Args:
            engagement_data (pd.DataFrame): DataFrame containing engagement data
            
        Returns:
            dict: Detection results with anomaly scores and classifications
        """
        # Extract features
        features = self.extract_engagement_features(engagement_data)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Detect anomalies based on method
        if self.detection_method == 'isolation_forest':
            predictions = self.model.fit_predict(features_scaled)
            anomaly_scores = self.model.decision_function(features_scaled)
            is_anomaly = predictions == -1
            
        elif self.detection_method == 'dbscan':
            predictions = self.model.fit_predict(features_scaled)
            is_anomaly = predictions == -1
            anomaly_scores = np.where(is_anomaly, -1.0, 1.0)
            
        else:  # random_forest (requires labeled data)
            if not self.is_trained:
                raise ValueError("Random Forest requires training with labeled data")
            predictions = self.model.predict(features_scaled)
            is_anomaly = predictions == 1
            anomaly_scores = self.model.predict_proba(features_scaled)[:, 1]
        
        # Create results
        results = pd.DataFrame({
            'index': engagement_data.index,
            'is_fake_engagement': is_anomaly,
            'anomaly_score': anomaly_scores,
            'detection_method': self.detection_method
        })
        
        # Calculate summary statistics
        fake_count = sum(is_anomaly)
        total_count = len(is_anomaly)
        fake_percentage = (fake_count / total_count) * 100
        
        return {
            'results': results,
            'summary': {
                'total_engagements': total_count,
                'fake_engagements': fake_count,
                'fake_percentage': fake_percentage,
                'detection_method': self.detection_method
            },
            'feature_importance': self._get_feature_importance(features.columns)
        }
    
    def train_supervised(self, engagement_data, labels):
        """
        Train the model with labeled data (for supervised methods)
        
        Args:
            engagement_data (pd.DataFrame): Engagement data
            labels (array): Labels (0: real, 1: fake)
        """
        features = self.extract_engagement_features(engagement_data)
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    
    def _has_emoji(self, text):
        """Check if text contains emojis"""
        if pd.isna(text):
            return 0
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE
        )
        return 1 if emoji_pattern.search(text) else 0
    
    def _is_repetitive(self, text):
        """Check if text is repetitive (potential spam)"""
        if pd.isna(text) or len(text) < 10:
            return 0
        
        words = text.lower().split()
        if len(words) < 3:
            return 0
            
        # Check for repeated words
        unique_words = set(words)
        repetition_ratio = 1 - (len(unique_words) / len(words))
        
        return 1 if repetition_ratio > 0.5 else 0
    
    def _get_feature_importance(self, feature_names):
        """Get feature importance from the model"""
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, self.model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        return {}
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'detection_method': self.detection_method,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.detection_method = model_data['detection_method']
        self.is_trained = model_data['is_trained']
