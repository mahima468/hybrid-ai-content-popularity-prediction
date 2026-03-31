"""
IMDB Sentiment Analysis Model
A focused sentiment analysis module using IMDB reviews dataset
Uses Logistic Regression with TF-IDF vectorization and NLTK preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import joblib
import re
import os
from typing import List, Tuple, Dict, Any

# Download required NLTK data (only needs to be done once)
def download_nltk_data():
    """Download required NLTK data if not already present"""
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)

class IMDBSentimentModel:
    """
    IMDB Sentiment Analysis Model
    Specifically designed for IMDB movie reviews sentiment classification
    """
    
    def __init__(self):
        """Initialize the IMDB sentiment model"""
        self.vectorizer = None
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.is_trained = False
        
        # Download NLTK data
        download_nltk_data()
    
    def preprocess_text(self, text: str) -> str:
        """
        Comprehensive text preprocessing for IMDB reviews
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Cleaned and preprocessed text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags (common in IMDB reviews)
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and numbers (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remove stopwords and lemmatize
        filtered_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                # Lemmatize the token
                try:
                    lemmatized_token = self.lemmatizer.lemmatize(token)
                    filtered_tokens.append(lemmatized_token)
                except:
                    filtered_tokens.append(token)
        
        return ' '.join(filtered_tokens)
    
    def load_imdb_data(self, file_path: str = None, sample_size: int = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load IMDB dataset
        
        Args:
            file_path (str): Path to IMDB dataset CSV file
            sample_size (int): Number of samples to use (for testing)
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and labels
        """
        if file_path and os.path.exists(file_path):
            # Load from custom file
            df = pd.read_csv(file_path)
            if 'review' in df.columns and 'sentiment' in df.columns:
                texts = df['review']
                labels = df['sentiment'].map({'positive': 1, 'negative': 0})
            else:
                raise ValueError("CSV must contain 'review' and 'sentiment' columns")
        else:
            # Generate sample IMDB-style data for demonstration
            print("Generating sample IMDB-style data...")
            texts, labels = self._generate_sample_data(sample_size or 1000)
            texts = pd.Series(texts)
            labels = pd.Series(labels)
        
        return texts, labels
    
    def _generate_sample_data(self, num_samples: int) -> Tuple[List[str], List[int]]:
        """Generate sample IMDB-style reviews for testing"""
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
            "Marvelous movie that exceeded all expectations."
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
            "Terrible experience. Complete waste of time."
        ]
        
        texts = []
        labels = []
        
        for i in range(num_samples):
            if i % 2 == 0:
                # Add some variation to positive reviews
                base_review = np.random.choice(positive_reviews)
                review = f"{base_review} {'Really enjoyed it!' if np.random.random() > 0.5 else 'Loved every minute!'}"
                texts.append(review)
                labels.append(1)  # Positive
            else:
                # Add some variation to negative reviews
                base_review = np.random.choice(negative_reviews)
                review = f"{base_review} {'Completely disappointed!' if np.random.random() > 0.5 else 'Regret watching this!'}"
                texts.append(review)
                labels.append(0)  # Negative
        
        return texts, labels
    
    def train(self, texts: pd.Series, labels: pd.Series, test_size: float = 0.2, 
              max_features: int = 5000, random_state: int = 42) -> Dict[str, Any]:
        """
        Train the sentiment analysis model
        
        Args:
            texts (pd.Series): Training texts
            labels (pd.Series): Training labels (0=negative, 1=positive)
            test_size (float): Proportion of data for testing
            max_features (int): Maximum number of features for TF-IDF
            random_state (int): Random state for reproducibility
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        print("Preprocessing text data...")
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        print("Splitting data...")
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        print("Creating TF-IDF features...")
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8,  # Ignore terms that appear in more than 80% of documents
            stop_words='english'
        )
        
        # Fit and transform training data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print("Training Logistic Regression model...")
        # Train Logistic Regression model
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            C=1.0,
            solver='liblinear'
        )
        
        self.model.fit(X_train_tfidf, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_tfidf)
        y_test_pred = self.model.predict(X_test_tfidf)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Get classification report
        class_report = classification_report(y_test, y_test_pred, 
                                         target_names=['Negative', 'Positive'],
                                         output_dict=True)
        
        # Get confusion matrix
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        
        self.is_trained = True
        
        # Get feature importance (coefficients for Logistic Regression)
        feature_names = self.vectorizer.get_feature_names_out()
        feature_importance = dict(zip(feature_names, self.model.coef_[0]))
        
        # Sort features by importance
        top_positive_features = sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True)[:10]
        top_negative_features = sorted(feature_importance.items(), 
                                      key=lambda x: x[1])[:10]
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'feature_importance': feature_importance,
            'top_positive_features': top_positive_features,
            'top_negative_features': top_negative_features,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'vocabulary_size': len(feature_names)
        }
        
        print(f"Training completed! Test Accuracy: {test_accuracy:.4f}")
        return results
    
    def predict(self, texts: List[str], return_probabilities: bool = False) -> List[Dict[str, Any]]:
        """
        Predict sentiment for new texts
        
        Args:
            texts (List[str]): List of texts to predict
            return_probabilities (bool): Whether to return prediction probabilities
            
        Returns:
            List[Dict[str, Any]]: Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize texts
        X_tfidf = self.vectorizer.transform(processed_texts)
        
        # Make predictions
        predictions = self.model.predict(X_tfidf)
        
        results = []
        for i, text in enumerate(texts):
            result = {
                'text': text,
                'sentiment': 'Positive' if predictions[i] == 1 else 'Negative',
                'sentiment_code': int(predictions[i])
            }
            
            if return_probabilities:
                probabilities = self.model.predict_proba(X_tfidf[i])
                result['confidence'] = float(max(probabilities[0]))
                result['probabilities'] = {
                    'negative': float(probabilities[0][0]),
                    'positive': float(probabilities[0][1])
                }
            
            results.append(result)
        
        return results
    
    def predict_single(self, text: str, return_probabilities: bool = False) -> Dict[str, Any]:
        """
        Predict sentiment for a single text
        
        Args:
            text (str): Text to predict
            return_probabilities (bool): Whether to return prediction probabilities
            
        Returns:
            Dict[str, Any]: Prediction result
        """
        result = self.predict([text], return_probabilities)
        return result[0]
    
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
            'vectorizer': self.vectorizer,
            'is_trained': self.is_trained,
            'model_type': 'IMDB_Logistic_Regression'
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(model_data, filepath)
        print(f"Model saved successfully to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model using joblib
        
        Args:
            filepath (str): Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded successfully from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model
        
        Returns:
            Dict[str, Any]: Model information
        """
        if not self.is_trained:
            return {'status': 'Model not trained'}
        
        info = {
            'model_type': 'Logistic Regression',
            'vectorizer_type': 'TF-IDF',
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'max_features': self.vectorizer.max_features,
            'ngram_range': self.vectorizer.ngram_range,
            'is_trained': self.is_trained
        }
        
        return info


# Example usage and demonstration
def main():
    """
    Example usage of the IMDB Sentiment Model
    """
    print("=== IMDB Sentiment Analysis Model Demo ===\n")
    
    # Initialize the model
    sentiment_model = IMDBSentimentModel()
    
    # Load sample data
    print("1. Loading IMDB data...")
    texts, labels = sentiment_model.load_imdb_data(sample_size=1000)
    print(f"Loaded {len(texts)} reviews")
    print(f"Positive reviews: {sum(labels)}")
    print(f"Negative reviews: {len(labels) - sum(labels)}\n")
    
    # Train the model
    print("2. Training the model...")
    training_results = sentiment_model.train(texts, labels)
    
    print(f"Training Accuracy: {training_results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {training_results['test_accuracy']:.4f}")
    print(f"Vocabulary Size: {training_results['vocabulary_size']}\n")
    
    # Show top features
    print("3. Top Positive Features:")
    for feature, score in training_results['top_positive_features'][:5]:
        print(f"  {feature}: {score:.4f}")
    
    print("\nTop Negative Features:")
    for feature, score in training_results['top_negative_features'][:5]:
        print(f"  {feature}: {score:.4f}\n")
    
    # Test predictions
    print("4. Testing predictions...")
    test_reviews = [
        "This movie was absolutely amazing! I loved every minute of it.",
        "Terrible film! Complete waste of time and money.",
        "The acting was okay but the plot was confusing.",
        "Brilliant cinematography and outstanding performances!"
    ]
    
    for review in test_reviews:
        result = sentiment_model.predict_single(review, return_probabilities=True)
        print(f"Review: {review[:50]}...")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result.get('confidence', 'N/A'):.4f})")
        print()
    
    # Save the model
    print("5. Saving the model...")
    sentiment_model.save_model("models/imdb_sentiment_model.joblib")
    
    # Demonstrate loading
    print("6. Loading the model...")
    new_model = IMDBSentimentModel()
    new_model.load_model("models/imdb_sentiment_model.joblib")
    
    # Test with loaded model
    result = new_model.predict_single("Great movie with excellent acting!", return_probabilities=True)
    print(f"Loaded model prediction: {result['sentiment']} (Confidence: {result.get('confidence', 'N/A'):.4f})")
    
    print("\n=== Demo completed successfully! ===")


if __name__ == "__main__":
    main()
