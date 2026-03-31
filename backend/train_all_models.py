"""
Complete Model Training Script
Trains all models for the Hybrid AI Content Popularity Prediction System
"""

import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def main():
    """
    Main training function that trains all models
    """
    print("=" * 80)
    print("HYBRID AI CONTENT POPULARITY PREDICTION SYSTEM")
    print("COMPLETE MODEL TRAINING")
    print("=" * 80)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create saved_models directory
    os.makedirs('saved_models', exist_ok=True)
    
    print("\n🔥 Training using REAL datasets...")
    print("🔥 All models will be trained on actual YouTube and Twitter data")
    
    # Train sentiment model
    print("\n" + "=" * 60)
    print("1. TRAINING SENTIMENT ANALYSIS MODEL")
    print("=" * 60)
    
    try:
        from train_sentiment_model import main as train_sentiment_main
        train_sentiment_main()
        print("✅ Sentiment model training completed successfully!")
    except Exception as e:
        print(f"❌ Error training sentiment model: {e}")
        return False
    
    # Train popularity prediction model
    print("\n" + "=" * 60)
    print("2. TRAINING POPULARITY PREDICTION MODEL")
    print("=" * 60)
    
    try:
        from train_prediction_model import main as train_popularity_main
        train_popularity_main()
        print("✅ Popularity prediction model training completed successfully!")
    except Exception as e:
        print(f"❌ Error training popularity prediction model: {e}")
        return False
    
    # Test model loading
    print("\n" + "=" * 60)
    print("3. TESTING MODEL LOADING")
    print("=" * 60)
    
    try:
        from load_trained_models import ModelLoader
        loader = ModelLoader()
        results = loader.load_all_models()
        
        if results['sentiment'] and results['popularity']:
            print("✅ All models loaded successfully!")
        else:
            print("⚠️  Some models failed to load")
            print(f"   Sentiment: {'✅' if results['sentiment'] else '❌'}")
            print(f"   Popularity: {'✅' if results['popularity'] else '❌'}")
    except Exception as e:
        print(f"❌ Error testing model loading: {e}")
        return False
    
    # Test predictions
    print("\n" + "=" * 60)
    print("4. TESTING PREDICTIONS")
    print("=" * 60)
    
    try:
        # Test sentiment prediction
        if loader.sentiment_model is not None:
            test_text = "I love this product! It's amazing!"
            result = loader.get_sentiment_prediction(test_text)
            print(f"✅ Sentiment test passed!")
            print(f"   Text: '{test_text}'")
            print(f"   Sentiment: {result.get('sentiment_name', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
        
        # Test popularity prediction
        if loader.popularity_model is not None:
            test_features = {
                'views': 10000,
                'likes': 500,
                'comments': 50,
                'engagement_rate': 5.5,
                'sentiment_score': 0.8
            }
            result = loader.get_popularity_prediction(test_features)
            print(f"✅ Popularity test passed!")
            print(f"   Current Views: {test_features['views']:,}")
            print(f"   Predicted Future Views: {result.get('predicted_future_views', 0):,}")
            print(f"   Confidence Interval: {result.get('confidence_interval', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error testing predictions: {e}")
        return False
    
    # List all saved models
    print("\n" + "=" * 60)
    print("5. SAVED MODELS SUMMARY")
    print("=" * 60)
    
    try:
        import glob
        model_files = glob.glob('saved_models/*')
        
        if model_files:
            print(f"Total files in saved_models: {len(model_files)}")
            for file_path in sorted(model_files):
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                print(f"  📄 {file_name} ({file_size:,} bytes)")
        else:
            print("❌ No model files found in saved_models directory")
            
    except Exception as e:
        print(f"❌ Error listing models: {e}")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📋 Next Steps:")
    print("1. Start the API server: uvicorn main_app:app --reload")
    print("2. Visit http://localhost:8000/docs to test the API")
    print("3. Check the /health endpoint to verify model loading")
    print("4. Use the frontend to test predictions")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All models trained and tested successfully!")
        sys.exit(0)
    else:
        print("\n💥 Training failed. Please check the errors above.")
        sys.exit(1)
