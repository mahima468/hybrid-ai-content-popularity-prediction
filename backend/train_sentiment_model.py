"""
Sentiment Model Training Script
Trains a sentiment analysis model using TextBlob and NLTK features
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SentimentModelTrainer:
    """
    Sentiment Model Training Class
    Handles data preparation, model training, and evaluation
    """
    
    def __init__(self):
        """Initialize the trainer with NLTK components"""
        self.lemmatizer = WordNetLemmatizer()
        self.model = None
        self.vectorizer = None
        self.training_data = None
        
        # Download NLTK data
        self._download_nltk_data()
        
        # Create saved models directory
        os.makedirs('saved_models', exist_ok=True)
        
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
    
    def clean_text(self, text):
        """
        Clean and preprocess text data
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remove stopwords and lemmatize
        stop_words = set(stopwords.words('english'))
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def get_textblob_sentiment(self, text):
        """
        Get sentiment using TextBlob
        
        Args:
            text (str): Input text
            
        Returns:
            int: Sentiment label (0=negative, 1=neutral, 2=positive)
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return 2  # Positive
            elif polarity < -0.1:
                return 0  # Negative
            else:
                return 1  # Neutral
        except:
            return 1  # Default to neutral
    
    def generate_sample_dataset(self, n_samples=1000):
        """
        Generate a synthetic dataset of social media comments
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Generated dataset
        """
        print(f"Generating {n_samples} sample social media comments...")
        
        np.random.seed(42)
        
        # Sample comments for each sentiment
        positive_comments = [
            "This is amazing! Love it so much! 🎉",
            "Absolutely fantastic! Best thing ever! ❤️",
            "Incredible work! So impressed! 👏",
            "Perfect! Exactly what I needed! ⭐",
            "Outstanding! Keep up the great work! 🌟",
            "Brilliant! Can't wait for more! 🚀",
            "Excellent! Highly recommend! 👍",
            "Wonderful! Made my day! 😊",
            "Superb! Quality is amazing! 💯",
            "Fantastic! Beyond expectations! 🎊",
            "Great job! Really appreciate this! 🙏",
            "Love it! So helpful and useful! 💖",
            "Awesome! Best decision ever! 🔥",
            "Perfect! Couldn't be happier! 🌈",
            "Amazing! Changed everything! ✨"
        ]
        
        negative_comments = [
            "This is terrible! Hate it! 😡",
            "Awful! Worst experience ever! 👎",
            "Disappointing! So bad! 💔",
            "Horrible! Waste of time! 🗑️",
            "Terrible! Never again! 🚫",
            "Awful! Completely useless! ❌",
            "Disgusting! So disappointed! 😤",
            "Worst! Don't recommend! 🚩",
            "Terrible! Total disaster! 💣",
            "Horrible! Regret this! 😞",
            "Awful! Poor quality! 📉",
            "Terrible! Not worth it! 💸",
            "Disgusting! So frustrating! 😫",
            "Worst! Complete failure! 🏴",
            "Horrible! So angry! 😠"
        ]
        
        neutral_comments = [
            "It's okay, nothing special.",
            "Average experience, could be better.",
            "It's fine, meets expectations.",
            "Not bad, but not great either.",
            "Decent, room for improvement.",
            "It's alright, nothing amazing.",
            "Acceptable, could use some work.",
            "Fair enough, does the job.",
            "It's okay, meets basic needs.",
            "Average, nothing outstanding.",
            "It's fine, nothing to complain about.",
            "Decent quality for the price.",
            "Acceptable, meets requirements.",
            "It's alright, could be improved.",
            "Fair, does what it's supposed to do."
        ]
        
        # Generate data
        data = []
        sentiments = []
        
        for i in range(n_samples):
            sentiment_choice = np.random.random()
            
            if sentiment_choice < 0.4:  # 40% positive
                comment = np.random.choice(positive_comments)
                sentiment = 2
            elif sentiment_choice < 0.8:  # 40% negative
                comment = np.random.choice(negative_comments)
                sentiment = 0
            else:  # 20% neutral
                comment = np.random.choice(neutral_comments)
                sentiment = 1
            
            # Add some variation
            if np.random.random() > 0.7:
                extra_words = np.random.choice([
                    " really", " very", " so", " quite", " extremely",
                    " somewhat", " incredibly", " moderately"
                ])
                comment = comment.replace("!", f" {extra_words}!")
            
            data.append(comment)
            sentiments.append(sentiment)
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': data,
            'sentiment': sentiments,
            'sentiment_label': ['negative' if s == 0 else 'neutral' if s == 1 else 'positive' for s in sentiments]
        })
        
        # Add TextBlob sentiment as additional feature
        df['textblob_sentiment'] = df['text'].apply(self.get_textblob_sentiment)
        
        # Clean text
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Add text features
        df['text_length'] = df['text'].str.len()
        df['cleaned_length'] = df['cleaned_text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        
        print(f"Generated dataset with {len(df)} samples")
        print(f"Positive: {len(df[df['sentiment'] == 2])}")
        print(f"Neutral: {len(df[df['sentiment'] == 1])}")
        print(f"Negative: {len(df[df['sentiment'] == 0])}")
        
        return df
    
    def load_real_dataset(self, file_path=None):
        """
        Load real Twitter sentiment dataset
        
        Args:
            file_path (str): Path to Twitter sentiment dataset
            
        Returns:
            pd.DataFrame: Dataset with mapped sentiment labels
        """
        print("=" * 60)
        print("Training using REAL datasets...")
        print("=" * 60)
        
        # Default to processed Twitter sentiment data
        if file_path is None:
            file_path = "../datasets/processed/sentiment_training_data.csv"
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found: {file_path}")
        
        print(f"Loading real dataset from: {file_path}")
        df = pd.read_csv(file_path)
        
        print(f"Real dataset loaded successfully")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Map sentiment labels from Twitter format to our format
        # Twitter: 0=negative, 2=neutral, 4=positive
        # Our format: negative=-1, neutral=0, positive=1
        if 'sentiment_label' in df.columns:
            # Map original Twitter labels
            label_mapping = {0: -1, 2: 0, 4: 1}
            df['sentiment'] = df['sentiment_label'].map(label_mapping)
        elif 'sentiment' in df.columns:
            # If already has sentiment names, map to scores
            sentiment_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
            df['sentiment'] = df['sentiment'].map(sentiment_mapping)
        else:
            raise ValueError("Dataset must contain 'sentiment_label' or 'sentiment' column")
        
        # Ensure we have cleaned text
        if 'cleaned_text' not in df.columns:
            if 'text' in df.columns:
                df['cleaned_text'] = df['text'].apply(self.clean_text)
            else:
                raise ValueError("Dataset must contain 'text' column")
        
        # Remove any rows with missing sentiment or text
        initial_count = len(df)
        df = df.dropna(subset=['sentiment', 'cleaned_text'])
        df = df[df['cleaned_text'].str.len() > 0]
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            print(f"Removed {removed_count} rows with missing data")
        
        # Add text features if not present
        if 'text_length' not in df.columns:
            df['text_length'] = df['text'].str.len()
        if 'word_count' not in df.columns:
            df['word_count'] = df['cleaned_text'].str.split().str.len()
        
        # Print sentiment distribution
        print("\n=== Real Dataset Sentiment Distribution ===")
        sentiment_counts = df['sentiment'].value_counts().sort_index()
        for sentiment, count in sentiment_counts.items():
            label_name = {1: 'positive', 0: 'neutral', -1: 'negative'}.get(sentiment, 'unknown')
            percentage = (count / len(df)) * 100
            print(f"{label_name.capitalize()} ({sentiment}): {count} ({percentage:.1f}%)")
        
        print(f"Final dataset shape: {df.shape}")
        print("Real dataset loaded successfully ✓")
        
        self.training_data = df
        return df
    
    def train_model(self, df, model_type='random_forest', test_size=0.2):
        """
        Train sentiment analysis model
        
        Args:
            df (pd.DataFrame): Training dataset
            model_type (str): Type of model to train
            test_size (float): Test set size
            
        Returns:
            dict: Training results
        """
        print(f"Training {model_type} sentiment model...")
        
        # Prepare features and target
        X = df['cleaned_text']
        y = df['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Create pipeline
        if model_type == 'random_forest':
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic_regression':
            classifier = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'naive_bayes':
            classifier = MultinomialNB()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create pipeline with TF-IDF vectorizer
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', classifier)
        ])
        
        # Train model
        print("Training model...")
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"\nTraining Results:")
        print(f"Model Type: {model_type}")
        print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_test_pred, 
                                 target_names=['negative', 'neutral', 'positive']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Store model and vectorizer
        self.model = pipeline
        self.vectorizer = pipeline.named_steps['tfidf']
        
        return {
            'model_type': model_type,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': classification_report(y_test, y_test_pred),
            'confusion_matrix': cm,
            'feature_count': len(self.vectorizer.get_feature_names_out())
        }
    
    def save_model(self, model_name='sentiment_model'):
        """
        Save trained model and components
        
        Args:
            model_name (str): Name for the model files
        """
        if self.model is None:
            raise ValueError("No trained model to save")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = f'saved_models/{model_name}_{timestamp}.joblib'
        joblib.dump(self.model, model_path)
        
        # Save vectorizer separately
        vectorizer_path = f'saved_models/{model_name}_vectorizer_{timestamp}.joblib'
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Save metadata
        metadata = {
            'model_type': 'sentiment_analysis',
            'timestamp': timestamp,
            'model_path': model_path,
            'vectorizer_path': vectorizer_path,
            'training_samples': len(self.training_data) if self.training_data is not None else 0
        }
        
        metadata_path = f'saved_models/{model_name}_metadata_{timestamp}.json'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nModel saved successfully!")
        print(f"Model: {model_path}")
        print(f"Vectorizer: {vectorizer_path}")
        print(f"Metadata: {metadata_path}")
        
        return metadata
    
    def predict_sentiment(self, text):
        """
        Predict sentiment for new text
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            raise ValueError("No trained model available")
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Make prediction
        sentiment_label = self.model.predict([cleaned_text])[0]
        sentiment_proba = self.model.predict_proba([cleaned_text])[0]
        
        # Map label to sentiment name
        sentiment_names = {0: 'negative', 1: 'neutral', 2: 'positive'}
        sentiment_name = sentiment_names.get(sentiment_label, 'unknown')
        
        # Get TextBlob sentiment for comparison
        textblob_sentiment = self.get_textblob_sentiment(text)
        textblob_name = sentiment_names.get(textblob_sentiment, 'unknown')
        
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'sentiment_label': sentiment_label,
            'sentiment_name': sentiment_name,
            'confidence': max(sentiment_proba),
            'probabilities': {
                'negative': sentiment_proba[0],
                'neutral': sentiment_proba[1],
                'positive': sentiment_proba[2]
            },
            'textblob_sentiment': textblob_name
        }


def main():
    """
    Main training function
    """
    print("=" * 60)
    print("SENTIMENT MODEL TRAINING")
    print("=" * 60)
    
    # Initialize trainer
    trainer = SentimentModelTrainer()
    
    try:
        # Load real dataset
        df = trainer.load_real_dataset()
        
        print(f"\nDataset Info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample text: {df['text'].iloc[0]}")
        print(f"Sample cleaned text: {df['cleaned_text'].iloc[0]}")
        print(f"Sample sentiment: {df['sentiment'].iloc[0]}")
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure the Twitter sentiment dataset is available at:")
        print("  ../datasets/processed/sentiment_training_data.csv")
        print("Or run the data preprocessing script first.")
        return
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Train multiple models
    models_to_train = ['random_forest', 'logistic_regression', 'naive_bayes']
    results = {}
    
    for model_type in models_to_train:
        print(f"\n{'-' * 40}")
        result = trainer.train_model(df, model_type=model_type)
        results[model_type] = result
        
        # Save the model
        metadata = trainer.save_model(f'sentiment_{model_type}')
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 60}")
    
    for model_type, result in results.items():
        print(f"\n{model_type.upper()}:")
        print(f"  Training Accuracy: {result['train_accuracy']:.4f} ({result['train_accuracy']*100:.2f}%)")
        print(f"  Test Accuracy: {result['test_accuracy']:.4f} ({result['test_accuracy']*100:.2f}%)")
        print(f"  Features: {result['feature_count']}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"\n🏆 BEST MODEL: {best_model[0].upper()}")
    print(f"   Test Accuracy: {best_model[1]['test_accuracy']:.4f} ({best_model[1]['test_accuracy']*100:.2f}%)")
    
    # Test with sample predictions
    print(f"\n{'=' * 60}")
    print("SAMPLE PREDICTIONS")
    print(f"{'=' * 60}")
    
    # Load the best model for testing
    trainer.train_model(df, model_type=best_model[0])
    
    sample_texts = [
        "I love this product! It's amazing! 🎉",
        "This is terrible and I hate it! 😡",
        "It's okay, nothing special.",
        "Fantastic work! Keep it up! 👏",
        "Worst experience ever! 🚫"
    ]
    
    for text in sample_texts:
        prediction = trainer.predict_sentiment(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {prediction['sentiment_name']} (confidence: {prediction['confidence']:.3f})")
        print(f"Probabilities: {prediction['probabilities']}")
    
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
