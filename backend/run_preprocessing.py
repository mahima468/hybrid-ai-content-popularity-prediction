#!/usr/bin/env python3
"""
Run dataset preprocessing standalone
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Run preprocessing"""
    print("Starting dataset preprocessing...")
    
    try:
        from utils.dataset_preprocessor import preprocess_dataset
        
        # Run preprocessing
        df = preprocess_dataset()
        
        print(f"✅ Preprocessing completed successfully!")
        print(f"   Dataset shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Has 'future_views': {'future_views' in df.columns}")
        print(f"   Has 'views': {'views' in df.columns}")
        print(f"   Null values in 'views': {df['views'].isnull().sum()}")
        print(f"   Null values in 'future_views': {df['future_views'].isnull().sum()}")
        
        # Show sample data
        print("\nSample data:")
        print(df[['views', 'future_views']].head(3))
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
