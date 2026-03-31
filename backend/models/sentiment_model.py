"""
Sentiment Analysis Model
Handles sentiment analysis using NLP techniques for content sentiment classification
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import joblib
import re

class SentimentModel:
    """
    Sentiment Analysis Model Class
    Provides sentiment classification for text content using multiple ML approaches
    """
    
    def __init__(self, model_type='naive_bayes'):
        """
        Initialize the sentiment model
        
        Args:
            model_type (str): Type of model to use ('naive_bayes', 'logistic', 'random_forest')
        """
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = None
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'logistic':
            self.model = LogisticRegression(random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("Model type must be 'naive_bayes', 'logistic', or 'random_forest'")
    
    def preprocess_text(self, text):
        """
        Preprocess text data for sentiment analysis
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and remove stopwords
        try:
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
            text = ' '.join(tokens)
        except:
            # Fallback if NLTK data not available
            pass
            
        return text
    
    def train(self, texts, labels):
        """
        Train the sentiment model
        
        Args:
            texts (list): List of text samples
            labels (list): Corresponding sentiment labels (0: negative, 1: neutral, 2: positive)
        """
        # Preprocess texts
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize texts
        X = self.vectorizer.fit_transform(preprocessed_texts)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
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
    
    def predict(self, texts):
        """
        Predict sentiment for given texts
        
        Args:
            texts (list): List of text samples to predict
            
        Returns:
            list: Predicted sentiment labels and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess texts
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize
        X = self.vectorizer.transform(preprocessed_texts)
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        results = []
        for i, text in enumerate(texts):
            results.append({
                'text': text,
                'sentiment': int(predictions[i]),
                'sentiment_label': self._get_sentiment_label(predictions[i]),
                'confidence': float(max(probabilities[i]))
            })
        
        return results
    
    def _get_sentiment_label(self, sentiment_num):
        """Convert numeric sentiment to label"""
        labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
        return labels.get(sentiment_num, 'unknown')
    
    def get_textblob_sentiment(self, text):
        """
        Get sentiment using TextBlob as alternative method
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Sentiment analysis result
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify sentiment based on polarity
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'text': text,
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
