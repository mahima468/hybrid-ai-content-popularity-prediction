"""
Fake Engagement Detection Module
Specialized module for detecting fake or suspicious engagement patterns in social media content
Uses Isolation Forest for anomaly detection and calculates Engagement Authenticity Score
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score
import joblib
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class FakeEngagementDetector:
    """
    Fake Engagement Detection System
    Detects suspicious engagement patterns using Isolation Forest and calculates authenticity scores
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize the Fake Engagement Detector
        
        Args:
            contamination (float): Expected proportion of anomalies in the dataset
            random_state (int): Random state for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = RobustScaler()  # Robust to outliers
        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.threshold = None
        
        # Initialize Isolation Forest
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto',
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1
        )
    
    def extract_engagement_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive engagement features for anomaly detection
        
        Args:
            data (pd.DataFrame): DataFrame containing engagement data
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        features = data.copy()
        
        # Basic engagement metrics
        required_columns = ['views', 'likes', 'comments']
        for col in required_columns:
            if col not in features.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Handle missing values and zeros
        features['views'] = features['views'].fillna(0).replace(0, 1)
        features['likes'] = features['likes'].fillna(0)
        features['comments'] = features['comments'].fillna(0)
        
        # 1. Engagement Rate Calculations
        features['engagement_rate'] = (features['likes'] + features['comments']) / features['views']
        features['like_rate'] = features['likes'] / features['views']
        features['comment_rate'] = features['comments'] / features['views']
        
        # 2. Ratio Features
        features['likes_to_views_ratio'] = features['likes'] / features['views']
        features['comments_to_views_ratio'] = features['comments'] / features['views']
        features['comments_to_likes_ratio'] = features['comments'] / (features['likes'] + 1)
        features['likes_to_comments_ratio'] = features['likes'] / (features['comments'] + 1)
        
        # 3. Advanced Engagement Patterns
        # Engagement velocity (if timestamp available)
        if 'timestamp' in features.columns:
            features['timestamp'] = pd.to_datetime(features['timestamp'])
            features['hour_of_day'] = features['timestamp'].dt.hour
            features['day_of_week'] = features['timestamp'].dt.dayofweek
            features['is_weekend'] = (features['timestamp'].dt.dayofweek >= 5).astype(int)
            
            # Time-based engagement patterns
            features['engagement_velocity'] = features['engagement_rate'] * 24  # Normalize to 24-hour period
        
        # 4. Statistical Features
        # Log transformations to handle skewness
        features['log_views'] = np.log1p(features['views'])
        features['log_likes'] = np.log1p(features['likes'])
        features['log_comments'] = np.log1p(features['comments'])
        
        # 5. Quality Indicators
        # Comment quality (comments per like ratio - high ratio might indicate genuine engagement)
        features['comment_quality_score'] = features['comments_to_likes_ratio']
        
        # Virality indicators
        features['virality_potential'] = features['likes'] * features['comments']
        features['normalized_virality'] = features['virality_potential'] / (features['views'] + 1)
        
        # 6. Suspicious Pattern Indicators
        # Perfect ratios (suspicious if too perfect)
        features['perfect_ratio_indicator'] = (
            (features['likes_to_views_ratio'] == 0.01) | 
            (features['likes_to_views_ratio'] == 0.05) |
            (features['comments_to_views_ratio'] == 0.01)
        ).astype(int)
        
        # Round number indicators (suspicious if all numbers are round)
        features['round_numbers_count'] = (
            (features['views'] % 100 == 0).astype(int) +
            (features['likes'] % 100 == 0).astype(int) +
            (features['comments'] % 10 == 0).astype(int)
        )
        
        # 7. User Behavior Patterns (if user_id available)
        if 'user_id' in features.columns:
            # User engagement consistency
            user_stats = features.groupby('user_id').agg({
                'engagement_rate': ['mean', 'std'],
                'views': 'count'
            }).fillna(0)
            
            user_stats.columns = ['user_avg_engagement', 'user_std_engagement', 'user_content_count']
            user_stats['user_engagement_consistency'] = 1 / (user_stats['user_std_engagement'] + 1)
            
            # Merge back to main dataframe
            features = features.merge(user_stats, left_on='user_id', right_index=True, how='left')
            features['user_engagement_consistency'] = features['user_engagement_consistency'].fillna(0)
        
        # 8. Content-based Features (if content_id available)
        if 'content_id' in features.columns:
            # Content performance relative to average
            content_stats = features.groupby('content_id').agg({
                'engagement_rate': 'mean',
                'views': 'sum'
            })
            
            overall_avg_engagement = features['engagement_rate'].mean()
            features['content_engagement_deviation'] = features['engagement_rate'] - overall_avg_engagement
            features['content_performance_score'] = features['engagement_rate'] / (overall_avg_engagement + 0.001)
        
        # Store feature names for later use
        numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_names = [col for col in numeric_features if col not in ['views', 'likes', 'comments']]
        
        return features[self.feature_names]
    
    def train(self, data: pd.DataFrame, optimize_threshold: bool = True) -> Dict[str, Any]:
        """
        Train the Isolation Forest model on engagement data
        
        Args:
            data (pd.DataFrame): Training data with engagement metrics
            optimize_threshold (bool): Whether to optimize anomaly threshold
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        print("Extracting engagement features...")
        X = self.extract_engagement_features(data)
        
        print(f"Extracted {len(self.feature_names)} features")
        print(f"Training on {len(X)} samples...")
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        print("Training Isolation Forest...")
        self.model.fit(X_scaled)
        
        # Get anomaly scores
        anomaly_scores = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)
        
        # Calculate authenticity scores
        authenticity_scores = self._calculate_authenticity_scores(anomaly_scores)
        
        # Optimize threshold if requested
        if optimize_threshold:
            self.threshold = self._optimize_threshold(anomaly_scores)
        else:
            self.threshold = np.percentile(anomaly_scores, self.contamination * 100)
        
        self.is_trained = True
        
        # Calculate training metrics
        metrics = {
            'training_samples': len(X),
            'features_used': len(self.feature_names),
            'contamination_rate': self.contamination,
            'threshold': self.threshold,
            'anomaly_score_stats': {
                'mean': float(np.mean(anomaly_scores)),
                'std': float(np.std(anomaly_scores)),
                'min': float(np.min(anomaly_scores)),
                'max': float(np.max(anomaly_scores))
            },
            'authenticity_score_stats': {
                'mean': float(np.mean(authenticity_scores)),
                'std': float(np.std(authenticity_scores)),
                'min': float(np.min(authenticity_scores)),
                'max': float(np.max(authenticity_scores))
            },
            'feature_importance': self._get_feature_importance(X)
        }
        
        print(f"Training completed!")
        print(f"Anomaly threshold: {self.threshold:.4f}")
        print(f"Expected anomalies: {self.contamination * 100:.1f}%")
        
        return metrics
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fake engagement for new data
        
        Args:
            data (pd.DataFrame): Test data with engagement metrics
            
        Returns:
            pd.DataFrame: Results with authenticity scores and predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        print("Extracting features for prediction...")
        X = self.extract_engagement_features(data)
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and scores
        predictions = self.model.predict(X_scaled)
        anomaly_scores = self.model.decision_function(X_scaled)
        authenticity_scores = self._calculate_authenticity_scores(anomaly_scores)
        
        # Create results dataframe
        results = data.copy()
        results['is_fake_engagement'] = predictions == -1
        results['anomaly_score'] = anomaly_scores
        results['engagement_authenticity_score'] = authenticity_scores
        results['authenticity_level'] = self._get_authenticity_level(authenticity_scores)
        results['suspicion_level'] = self._get_suspicion_level(anomaly_scores)
        
        return results
    
    def _calculate_authenticity_scores(self, anomaly_scores: np.ndarray) -> np.ndarray:
        """
        Calculate Engagement Authenticity Score from anomaly scores
        
        Args:
            anomaly_scores (np.ndarray): Anomaly scores from Isolation Forest
            
        Returns:
            np.ndarray: Authenticity scores (0-100)
        """
        # Normalize scores to 0-100 scale
        # Higher anomaly score = more authentic
        min_score = np.min(anomaly_scores)
        max_score = np.max(anomaly_scores)
        
        if max_score == min_score:
            return np.full_like(anomaly_scores, 50.0)
        
        # Normalize to 0-100 scale
        normalized_scores = (anomaly_scores - min_score) / (max_score - min_score) * 100
        
        return normalized_scores
    
    def _get_authenticity_level(self, scores: np.ndarray) -> List[str]:
        """Convert authenticity scores to categorical levels"""
        levels = []
        for score in scores:
            if score >= 80:
                levels.append('Very High')
            elif score >= 60:
                levels.append('High')
            elif score >= 40:
                levels.append('Medium')
            elif score >= 20:
                levels.append('Low')
            else:
                levels.append('Very Low')
        return levels
    
    def _get_suspicion_level(self, anomaly_scores: np.ndarray) -> List[str]:
        """Convert anomaly scores to suspicion levels"""
        levels = []
        for score in anomaly_scores:
            if score <= self.threshold * 0.5:
                levels.append('Very High')
            elif score <= self.threshold:
                levels.append('High')
            elif score <= self.threshold * 1.5:
                levels.append('Medium')
            elif score <= self.threshold * 2:
                levels.append('Low')
            else:
                levels.append('Very Low')
        return levels
    
    def _optimize_threshold(self, anomaly_scores: np.ndarray) -> float:
        """
        Optimize anomaly threshold based on score distribution
        
        Args:
            anomaly_scores (np.ndarray): Anomaly scores
            
        Returns:
            float: Optimized threshold
        """
        # Use percentile-based approach with adjustment
        base_threshold = np.percentile(anomaly_scores, self.contamination * 100)
        
        # Adjust based on score distribution
        score_std = np.std(anomaly_scores)
        if score_std > 0.1:
            # High variance - be more conservative
            adjustment = score_std * 0.5
        else:
            # Low variance - use standard threshold
            adjustment = 0
        
        return base_threshold - adjustment
    
    def _get_feature_importance(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate feature importance based on variance and correlation
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            Dict[str, float]: Feature importance scores
        """
        importance = {}
        for feature in X.columns:
            # Calculate importance based on variance
            variance = X[feature].var()
            importance[feature] = variance
        
        # Normalize importance scores
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v/total_importance for k, v in importance.items()}
        
        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def analyze_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze engagement patterns and provide insights
        
        Args:
            data (pd.DataFrame): Engagement data
            
        Returns:
            Dict[str, Any]: Pattern analysis results
        """
        results = self.predict(data)
        
        # Calculate statistics
        fake_count = results['is_fake_engagement'].sum()
        total_count = len(results)
        fake_percentage = (fake_count / total_count) * 100
        
        # Authenticity score distribution
        authenticity_stats = {
            'mean': results['engagement_authenticity_score'].mean(),
            'median': results['engagement_authenticity_score'].median(),
            'std': results['engagement_authenticity_score'].std(),
            'q25': results['engagement_authenticity_score'].quantile(0.25),
            'q75': results['engagement_authenticity_score'].quantile(0.75)
        }
        
        # Suspicion level distribution
        suspicion_distribution = results['suspicion_level'].value_counts().to_dict()
        
        # Feature analysis for fake vs real engagement
        fake_engagement = results[results['is_fake_engagement'] == True]
        real_engagement = results[results['is_fake_engagement'] == False]
        
        comparison = {}
        for feature in ['engagement_rate', 'likes_to_views_ratio', 'comments_to_views_ratio']:
            if feature in data.columns:
                comparison[feature] = {
                    'fake_mean': fake_engagement[feature].mean() if len(fake_engagement) > 0 else 0,
                    'real_mean': real_engagement[feature].mean() if len(real_engagement) > 0 else 0,
                    'fake_std': fake_engagement[feature].std() if len(fake_engagement) > 0 else 0,
                    'real_std': real_engagement[feature].std() if len(real_engagement) > 0 else 0
                }
        
        return {
            'total_analyzed': total_count,
            'fake_engagement_count': fake_count,
            'fake_engagement_percentage': fake_percentage,
            'authenticity_score_statistics': authenticity_stats,
            'suspicion_level_distribution': suspicion_distribution,
            'feature_comparison': comparison,
            'threshold_used': self.threshold
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model using joblib
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'threshold': self.threshold,
            'contamination': self.contamination,
            'random_state': self.random_state,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved successfully to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model using joblib
        
        Args:
            filepath (str): Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.threshold = model_data['threshold']
        self.contamination = model_data['contamination']
        self.random_state = model_data['random_state']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded successfully from {filepath}")
    
    def generate_sample_data(self, n_samples: int = 1000, fake_ratio: float = 0.1) -> pd.DataFrame:
        """
        Generate sample engagement data for testing
        
        Args:
            n_samples (int): Number of samples to generate
            fake_ratio (float): Ratio of fake engagement samples
            
        Returns:
            pd.DataFrame: Sample engagement data
        """
        np.random.seed(self.random_state)
        
        # Generate realistic engagement data
        views = np.random.lognormal(mean=8, sigma=1.5, size=n_samples).astype(int)
        views = np.clip(views, 100, 1000000)
        
        # Realistic like rates (0.5% to 5%)
        like_rates = np.random.beta(2, 50, size=n_samples) * 0.05
        likes = (views * like_rates).astype(int)
        
        # Realistic comment rates (0.1% to 1%)
        comment_rates = np.random.beta(2, 100, size=n_samples) * 0.01
        comments = (views * comment_rates).astype(int)
        
        # Create DataFrame
        data = pd.DataFrame({
            'content_id': range(1, n_samples + 1),
            'user_id': np.random.randint(1, 100, n_samples),
            'views': views,
            'likes': likes,
            'comments': comments,
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='h')
        })
        
        # Add fake engagement patterns
        n_fake = int(n_samples * fake_ratio)
        fake_indices = np.random.choice(n_samples, n_fake, replace=False)
        
        for idx in fake_indices:
            # Fake engagement patterns
            if np.random.random() > 0.5:
                # High likes, low comments (bot-like)
                data.loc[idx, 'likes'] = int(data.loc[idx, 'views'] * 0.1)
                data.loc[idx, 'comments'] = np.random.randint(0, 5)
            else:
                # Perfect ratios (suspicious)
                data.loc[idx, 'likes'] = int(data.loc[idx, 'views'] * 0.05)
                data.loc[idx, 'comments'] = int(data.loc[idx, 'views'] * 0.01)
        
        return data


# Example usage and demonstration
def main():
    """
    Example usage of the Fake Engagement Detector
    """
    print("=== Fake Engagement Detection Demo ===\n")
    
    # Initialize the detector
    detector = FakeEngagementDetector(contamination=0.1)
    
    # Generate sample data
    print("1. Generating sample engagement data...")
    data = detector.generate_sample_data(n_samples=1000, fake_ratio=0.15)
    print(f"Generated {len(data)} samples")
    print(f"Sample data preview:\n{data.head()}\n")
    
    # Train the model
    print("2. Training the detector...")
    training_results = detector.train(data)
    print(f"Training completed!")
    print(f"Features used: {training_results['features_used']}")
    print(f"Anomaly threshold: {training_results['threshold']:.4f}\n")
    
    # Make predictions
    print("3. Detecting fake engagement...")
    results = detector.predict(data)
    
    # Show results
    fake_count = results['is_fake_engagement'].sum()
    print(f"Detected {fake_count} fake engagement patterns ({fake_count/len(results)*100:.1f}%)")
    
    # Show some examples
    fake_examples = results[results['is_fake_engagement'] == True].head(3)
    real_examples = results[results['is_fake_engagement'] == False].head(3)
    
    print("\nFake Engagement Examples:")
    for _, row in fake_examples.iterrows():
        print(f"  Content {row['content_id']}: Authenticity Score = {row['engagement_authenticity_score']:.1f}, "
              f"Suspicion Level = {row['suspicion_level']}")
    
    print("\nReal Engagement Examples:")
    for _, row in real_examples.iterrows():
        print(f"  Content {row['content_id']}: Authenticity Score = {row['engagement_authenticity_score']:.1f}, "
              f"Suspicion Level = {row['suspicion_level']}")
    
    # Analyze patterns
    print("\n4. Analyzing engagement patterns...")
    analysis = detector.analyze_patterns(data)
    print(f"Authenticity Score Statistics:")
    print(f"  Mean: {analysis['authenticity_score_statistics']['mean']:.2f}")
    print(f"  Median: {analysis['authenticity_score_statistics']['median']:.2f}")
    print(f"  Std: {analysis['authenticity_score_statistics']['std']:.2f}")
    
    # Save model
    print("\n5. Saving the model...")
    detector.save_model("models/fake_engagement_detector.joblib")
    
    # Test loading
    print("6. Loading the model...")
    new_detector = FakeEngagementDetector()
    new_detector.load_model("models/fake_engagement_detector.joblib")
    
    # Test with new data
    test_data = detector.generate_sample_data(n_samples=100, fake_ratio=0.2)
    test_results = new_detector.predict(test_data)
    test_fake = test_results['is_fake_engagement'].sum()
    print(f"Loaded model detected {test_fake} fake patterns in test data")
    
    print("\n=== Demo completed successfully! ===")


if __name__ == "__main__":
    main()
