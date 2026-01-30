#!/usr/bin/env python3
"""
Example: Using the preprocessing module in a data pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
from preprocessing import preprocess_marketing_data
from data_loader import load_from_file

def main():
    print("Marketing Mix Model - Data Preprocessing Pipeline")
    print("="*80)
    
    # Step 1: Load raw data
    print("\n[1] Loading raw data...")
    df = load_from_file('../data/raw/marketing_data.csv')
    print(f"✓ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Step 2: Preprocess
    print("\n[2] Running preprocessing pipeline...")
    df_processed, feature_sets = preprocess_marketing_data(df)
    
    # Step 3: Save processed data
    print("\n[3] Saving processed data...")
    df_processed.to_csv('../data/processed/marketing_data_processed.csv', index=False)
    df_processed.to_parquet('../data/processed/marketing_data_processed.parquet', index=False)
    print("✓ Saved to data/processed/")
    
    # Step 4: Display feature sets
    print("\n[4] Feature Sets for Modeling:")
    print("-"*80)
    for feature_type, columns in feature_sets.items():
        if isinstance(columns, list):
            print(f"\n{feature_type.upper()} ({len(columns)}):")
            for col in columns[:5]:  # Show first 5
                print(f"  - {col}")
            if len(columns) > 5:
                print(f"  ... and {len(columns) - 5} more")
        else:
            print(f"\n{feature_type.upper()}: {columns}")
    
    # Step 5: Data quality check
    print("\n[5] Data Quality Check:")
    print("-"*80)
    print(f"Shape: {df_processed.shape}")
    print(f"Missing values: {df_processed.isnull().sum().sum()}")
    print(f"Completeness: {round((1 - df_processed.isnull().sum().sum() / (df_processed.shape[0] * df_processed.shape[1])) * 100, 1)}%")
    print(f"Memory usage: {df_processed.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    print("\n✓ Pipeline complete! Data ready for modeling.")

if __name__ == "__main__":
    main()
