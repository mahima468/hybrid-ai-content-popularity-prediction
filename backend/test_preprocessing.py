#!/usr/bin/env python3
"""
Test script for dataset preprocessing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataset_preprocessor import preprocess_dataset, get_dataset_summary

def test_preprocessing():
    """Test the dataset preprocessing"""
    try:
        print("Testing dataset preprocessing...")
        
        # Get initial summary
        print("\n=== Initial Dataset Summary ===")
        summary = get_dataset_summary()
        print(f"Shape: {summary.get('shape')}")
        print(f"Has future_views: {summary.get('has_future_views')}")
        print(f"Has views: {summary.get('has_views')}")
        
        # Run preprocessing
        print("\n=== Running Preprocessing ===")
        df = preprocess_dataset()
        
        # Check results
        print("\n=== Post-Preprocessing Results ===")
        print(f"Dataset shape: {df.shape}")
        print(f"Future views column exists: {'future_views' in df.columns}")
        print(f"Views column exists: {'views' in df.columns}")
        print(f"Null values in views: {df['views'].isnull().sum()}")
        print(f"Null values in future_views: {df['future_views'].isnull().sum()}")
        
        # Show sample data
        print("\n=== Sample Data ===")
        print(df[['views', 'future_views', 'engagement_rate', 'sentiment_score']].head())
        
        print("\n✅ Preprocessing test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Preprocessing test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_preprocessing()
