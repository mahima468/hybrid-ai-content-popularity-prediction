"""
Content Popularity Prediction Model
Machine learning regression model to predict future content popularity using RandomForestRegressor
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class ContentPopularityPredictor:
    """
    Content Popularity Prediction System
    Predicts future content popularity using RandomForestRegressor
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize the Content Popularity Predictor
        
        Args:
            n_estimators (int): Number of trees in the random forest
            random_state (int): Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = RobustScaler()  # Robust to outliers
        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.feature_importance = {}
        
        # Initialize RandomForestRegressor with optimized parameters
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            n_jobs=-1,
            verbose=0
        )
    
    def preprocess_data(self, data: pd.DataFrame, target_column: str = 'future_views') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data for popularity prediction
        
        Args:
            data (pd.DataFrame): Raw data with engagement metrics
            target_column (str): Name of the target variable column
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Preprocessed features and target
        """
        features = data.copy()
        
        # Check required columns
        required_columns = ['views', 'likes', 'comments', 'sentiment_score', 'engagement_rate']
        for col in required_columns:
            if col not in features.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        if target_column not in features.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Handle missing values
        for col in required_columns + [target_column]:
            if col in features.columns:
                features[col] = features[col].fillna(features[col].median())
        
        # Extract target variable
        y = features[target_column]
        
        # Feature Engineering
        engineered_features = self._engineer_features(features)
        
        # Select only numeric features for modeling
        numeric_features = engineered_features.select_dtypes(include=[np.number])
        
        # Remove target column from features
        if target_column in numeric_features.columns:
            numeric_features = numeric_features.drop(columns=[target_column])
        
        self.feature_names = numeric_features.columns.tolist()
        
        return numeric_features, y
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features from existing data
        
        Args:
            data (pd.DataFrame): Original data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        features = data.copy()
        
        # 1. Basic Ratios
        features['likes_to_views_ratio'] = features['likes'] / (features['views'] + 1)
        features['comments_to_views_ratio'] = features['comments'] / (features['views'] + 1)
        features['comments_to_likes_ratio'] = features['comments'] / (features['likes'] + 1)
        
        # 2. Engagement Metrics
        features['total_engagement'] = features['likes'] + features['comments']
        features['engagement_intensity'] = features['total_engagement'] / (features['views'] + 1)
        
        # 3. Sentiment-Engagement Interaction
        features['sentiment_engagement_product'] = features['sentiment_score'] * features['engagement_rate']
        features['sentiment_likes_product'] = features['sentiment_score'] * features['likes_to_views_ratio']
        
        # 4. Log Transformations (to handle skewness)
        features['log_views'] = np.log1p(features['views'])
        features['log_likes'] = np.log1p(features['likes'])
        features['log_comments'] = np.log1p(features['comments'])
        features['log_total_engagement'] = np.log1p(features['total_engagement'])
        
        # 5. Polynomial Features (interaction terms)
        features['views_squared'] = features['views'] ** 2
        features['likes_squared'] = features['likes'] ** 2
        features['comments_squared'] = features['comments'] ** 2
        
        # 6. Temporal Features (if timestamp available)
        if 'timestamp' in features.columns:
            features['timestamp'] = pd.to_datetime(features['timestamp'])
            features['hour_of_day'] = features['timestamp'].dt.hour
            features['day_of_week'] = features['timestamp'].dt.dayofweek
            features['month'] = features['timestamp'].dt.month
            features['is_weekend'] = (features['timestamp'].dt.dayofweek >= 5).astype(int)
            
            # Cyclical encoding for temporal features
            features['hour_sin'] = np.sin(2 * np.pi * features['hour_of_day'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour_of_day'] / 24)
            features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # 7. Statistical Features
        # Engagement velocity indicators
        features['engagement_velocity'] = features['engagement_rate'] * 24  # Normalize to 24 hours
        
        # Content quality indicators
        features['content_quality_score'] = (features['sentiment_score'] * features['total_engagement']) / (features['views'] + 1)
        
        # Virality potential
        features['virality_score'] = features['likes'] * features['comments'] * features['sentiment_score']
        
        # 8. Normalization Features
        # Normalize engagement metrics by typical ranges
        features['normalized_likes'] = features['likes'] / (features['views'] + 1)
        features['normalized_comments'] = features['comments'] / (features['views'] + 1)
        features['normalized_engagement'] = features['total_engagement'] / (features['views'] + 1)
        
        # 9. Composite Features
        # Popularity index
        features['popularity_index'] = (
            features['likes'] * 0.4 + 
            features['comments'] * 0.3 + 
            features['views'] * 0.2 + 
            features['sentiment_score'] * 100 * 0.1
        )
        
        # Engagement efficiency
        features['engagement_efficiency'] = features['total_engagement'] / (features['views'].mean() + 1)
        
        # 10. Outlier Indicators
        # Flag unusually high engagement
        mean_engagement_rate = features['engagement_rate'].mean()
        std_engagement_rate = features['engagement_rate'].std()
        features['high_engagement_flag'] = (features['engagement_rate'] > mean_engagement_rate + 2 * std_engagement_rate).astype(int)
        
        # Flag high sentiment engagement
        features['high_sentiment_engagement'] = (features['sentiment_engagement_product'] > 
                                               features['sentiment_engagement_product'].quantile(0.9)).astype(int)
        
        return features
    
    def train(self, data: pd.DataFrame, target_column: str = 'future_views', 
              test_size: float = 0.2, optimize_hyperparameters: bool = False) -> Dict[str, Any]:
        """
        Train the RandomForestRegressor model
        
        Args:
            data (pd.DataFrame): Training data with features and target
            target_column (str): Name of the target variable column
            test_size (float): Proportion of data for testing
            optimize_hyperparameters (bool): Whether to optimize hyperparameters
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        print("Preprocessing data...")
        X, y = self.preprocess_data(data, target_column)
        
        print(f"Extracted {len(self.feature_names)} features")
        print(f"Training on {len(X)} samples...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Optimize hyperparameters if requested
        if optimize_hyperparameters:
            print("Optimizing hyperparameters...")
            self.model = self._optimize_hyperparameters(X_train_scaled, y_train)
        else:
            print("Training RandomForestRegressor...")
            self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        # Feature importance
        self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        self.is_trained = True
        
        # Prepare results
        results = {
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': len(self.feature_names),
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_mean_r2': cv_scores.mean(),
            'cv_std_r2': cv_scores.std(),
            'feature_importance': dict(sorted(self.feature_importance.items(), 
                                             key=lambda x: x[1], reverse=True)),
            'top_features': dict(sorted(self.feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)[:10])
        }
        
        print(f"Training completed!")
        print(f"Test R² Score: {test_r2:.4f}")
        print(f"Test MAE: {test_mae:.2f}")
        print(f"Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return results
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: pd.Series) -> RandomForestRegressor:
        """
        Optimize hyperparameters using GridSearchCV
        
        Args:
            X (np.ndarray): Training features
            y (pd.Series): Training target
            
        Returns:
            RandomForestRegressor: Optimized model
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=self.random_state),
            param_grid,
            cv=3,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def predict(self, data: pd.DataFrame, return_confidence: bool = False) -> pd.DataFrame:
        """
        Predict future popularity for new data
        
        Args:
            data (pd.DataFrame): Test data with engagement metrics
            return_confidence (bool): Whether to return prediction confidence intervals
            
        Returns:
            pd.DataFrame: Results with predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        print("Preprocessing prediction data...")
        X, _ = self.preprocess_data(data)
        
        # Ensure feature order matches training
        X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Create results dataframe
        results = data.copy()
        results['predicted_future_views'] = predictions
        
        if return_confidence:
            # Calculate prediction intervals using tree predictions
            tree_predictions = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
            
            # Calculate confidence intervals (95%)
            lower_bound = np.percentile(tree_predictions, 2.5, axis=0)
            upper_bound = np.percentile(tree_predictions, 97.5, axis=0)
            
            results['prediction_lower_bound'] = lower_bound
            results['prediction_upper_bound'] = upper_bound
            results['prediction_interval_width'] = upper_bound - lower_bound
            results['prediction_confidence'] = 1 - (results['prediction_interval_width'] / predictions)
        
        return results
    
    def predict_single(self, data: pd.DataFrame, return_confidence: bool = False) -> Dict[str, Any]:
        """
        Predict popularity for a single content item
        
        Args:
            data (pd.DataFrame): Single row of data
            return_confidence (bool): Whether to return confidence intervals
            
        Returns:
            Dict[str, Any]: Prediction result
        """
        result = self.predict(data, return_confidence)
        return result.iloc[0].to_dict()
    
    def evaluate_model(self, data: pd.DataFrame, target_column: str = 'future_views') -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            data (pd.DataFrame): Test data
            target_column (str): Target variable name
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        X, y = self.preprocess_data(data, target_column)
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate percentage errors
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        
        # Residual analysis
        residuals = y - y_pred
        
        evaluation = {
            'r2_score': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'feature_importance': self.feature_importance,
            'top_features': dict(sorted(self.feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)[:10])
        }
        
        return evaluation
    
    def get_feature_analysis(self) -> Dict[str, Any]:
        """
        Get detailed feature analysis
        
        Returns:
            Dict[str, Any]: Feature analysis results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Sort features by importance
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Categorize features
        feature_categories = {
            'engagement_features': [],
            'sentiment_features': [],
            'ratio_features': [],
            'temporal_features': [],
            'statistical_features': [],
            'other_features': []
        }
        
        for feature, importance in sorted_features:
            if any(keyword in feature.lower() for keyword in ['engagement', 'like', 'comment', 'view']):
                feature_categories['engagement_features'].append((feature, importance))
            elif 'sentiment' in feature.lower():
                feature_categories['sentiment_features'].append((feature, importance))
            elif 'ratio' in feature.lower():
                feature_categories['ratio_features'].append((feature, importance))
            elif any(keyword in feature.lower() for keyword in ['hour', 'day', 'month', 'time']):
                feature_categories['temporal_features'].append((feature, importance))
            elif any(keyword in feature.lower() for keyword in ['log', 'squared', 'normalized']):
                feature_categories['statistical_features'].append((feature, importance))
            else:
                feature_categories['other_features'].append((feature, importance))
        
        return {
            'total_features': len(self.feature_names),
            'top_10_features': dict(sorted_features[:10]),
            'bottom_10_features': dict(sorted_features[-10:]),
            'feature_categories': feature_categories,
            'feature_importance_distribution': {
                'mean': np.mean(list(self.feature_importance.values())),
                'std': np.std(list(self.feature_importance.values())),
                'min': np.min(list(self.feature_importance.values())),
                'max': np.max(list(self.feature_importance.values()))
            }
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
            'feature_importance': self.feature_importance,
            'n_estimators': self.n_estimators,
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
        self.feature_importance = model_data['feature_importance']
        self.n_estimators = model_data['n_estimators']
        self.random_state = model_data['random_state']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded successfully from {filepath}")
    
    def generate_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate sample content data for testing
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Sample data with engagement metrics
        """
        np.random.seed(self.random_state)
        
        # Generate realistic engagement data
        views = np.random.lognormal(mean=8, sigma=1.5, size=n_samples).astype(int)
        views = np.clip(views, 100, 1000000)
        
        # Realistic engagement rates
        like_rates = np.random.beta(2, 50, size=n_samples) * 0.05
        likes = (views * like_rates).astype(int)
        
        comment_rates = np.random.beta(2, 100, size=n_samples) * 0.01
        comments = (views * comment_rates).astype(int)
        
        # Sentiment scores (normalized -1 to 1)
        sentiment_scores = np.random.normal(0.2, 0.3, size=n_samples)
        sentiment_scores = np.clip(sentiment_scores, -1, 1)
        
        # Engagement rate
        engagement_rates = (likes + comments) / views
        
        # Future views (target variable) - based on current metrics with some growth
        growth_factor = np.random.normal(1.5, 0.3, size=n_samples)
        growth_factor = np.clip(growth_factor, 0.5, 3.0)
        
        # Future views influenced by current engagement and sentiment
        sentiment_boost = 1 + (sentiment_scores * 0.3)  # Positive sentiment boosts growth
        engagement_boost = 1 + (engagement_rates * 10)  # Higher engagement boosts growth
        
        future_views = (views * growth_factor * sentiment_boost * engagement_boost).astype(int)
        future_views = np.clip(future_views, 100, 5000000)
        
        # Create DataFrame
        data = pd.DataFrame({
            'content_id': range(1, n_samples + 1),
            'views': views,
            'likes': likes,
            'comments': comments,
            'sentiment_score': sentiment_scores,
            'engagement_rate': engagement_rates,
            'future_views': future_views,
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H')
        })
        
        return data


# Example usage and demonstration
def main():
    """
    Example usage of the Content Popularity Predictor
    """
    print("=== Content Popularity Prediction Demo ===\n")
    
    # Initialize the predictor
    predictor = ContentPopularityPredictor(n_estimators=100)
    
    # Generate sample data
    print("1. Generating sample content data...")
    data = predictor.generate_sample_data(n_samples=1000)
    print(f"Generated {len(data)} samples")
    print(f"Sample data preview:\n{data[['views', 'likes', 'comments', 'sentiment_score', 'engagement_rate', 'future_views']].head()}\n")
    
    # Train the model
    print("2. Training the popularity predictor...")
    training_results = predictor.train(data, optimize_hyperparameters=False)
    
    print(f"Training completed!")
    print(f"Features used: {training_results['features_used']}")
    print(f"Test R² Score: {training_results['test_r2']:.4f}")
    print(f"Test MAE: {training_results['test_mae']:.2f}\n")
    
    # Show feature importance
    print("3. Top 10 Important Features:")
    for feature, importance in list(training_results['top_features'].items())[:10]:
        print(f"  {feature}: {importance:.4f}")
    print()
    
    # Make predictions
    print("4. Making predictions on new data...")
    test_data = predictor.generate_sample_data(n_samples=100)
    predictions = predictor.predict(test_data, return_confidence=True)
    
    # Show some predictions
    print("Sample Predictions:")
    for i in range(5):
        row = predictions.iloc[i]
        print(f"  Content {row['content_id']}: "
              f"Current Views: {row['views']:,}, "
              f"Predicted Future Views: {row['predicted_future_views']:,}")
        if 'prediction_lower_bound' in row:
            print(f"    Confidence Interval: {row['prediction_lower_bound']:,} - {row['prediction_upper_bound']:,}")
    print()
    
    # Evaluate model
    print("5. Evaluating model performance...")
    evaluation = predictor.evaluate_model(data)
    print(f"R² Score: {evaluation['r2_score']:.4f}")
    print(f"RMSE: {evaluation['rmse']:.2f}")
    print(f"MAPE: {evaluation['mape']:.2f}%")
    print()
    
    # Feature analysis
    print("6. Feature analysis...")
    feature_analysis = predictor.get_feature_analysis()
    print(f"Total features: {feature_analysis['total_features']}")
    print(f"Feature importance mean: {feature_analysis['feature_importance_distribution']['mean']:.4f}")
    
    # Save model
    print("\n7. Saving the model...")
    predictor.save_model("models/content_popularity_predictor.joblib")
    
    # Test loading
    print("8. Loading the model...")
    new_predictor = ContentPopularityPredictor()
    new_predictor.load_model("models/content_popularity_predictor.joblib")
    
    # Test with loaded model
    test_sample = test_data.head(1)
    loaded_prediction = new_predictor.predict_single(test_sample, return_confidence=True)
    print(f"Loaded model prediction: {loaded_prediction['predicted_future_views']:,}")
    
    print("\n=== Demo completed successfully! ===")


if __name__ == "__main__":
    main()
