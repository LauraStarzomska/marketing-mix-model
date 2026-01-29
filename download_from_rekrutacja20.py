#!/usr/bin/env python
"""Download marketing data from rekrutacja-20"""
import subprocess
from pathlib import Path
from google.cloud import bigquery
from google.oauth2.credentials import Credentials

print("Downloading data from alterdata-rekrutacja-20...")

# Get access token from gcloud
result = subprocess.run(
    ['/opt/homebrew/bin/gcloud', 'auth', 'print-access-token'],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print(f"Error getting token: {result.stderr}")
    exit(1)

token = result.stdout.strip()
credentials = Credentials(token=token)
client = bigquery.Client(project="alterdata-rekrutacja-20", credentials=credentials)

try:
    # Download cost_revenue table
    print("Downloading cost_revenue table...")
    df = client.query("SELECT * FROM `alterdata-rekrutacja-20.marketing_data.cost_revenue`").result().to_dataframe()
    print(f"  ✓ cost_revenue: {len(df):,} rows, {len(df.columns)} columns")
    
    # Try promotions if it exists
    try:
        print("Downloading promotions table...")
        df_promotions = client.query("SELECT * FROM `alterdata-rekrutacja-20.marketing_data.promotions`").result().to_dataframe()
        print(f"  ✓ promotions: {len(df_promotions):,} rows, {len(df_promotions.columns)} columns")
        
        # Merge on common columns (geo and week)
        common_cols = set(df.columns) & set(df_promotions.columns)
        print(f"Common columns: {common_cols}")
        
        if len(common_cols) >= 2:
            # Try to merge on week and geo
            if 'week' in common_cols and 'geo' in common_cols:
                print(f"Merging on: week, geo")
                df = df.merge(df_promotions, on=['week', 'geo'], how='left')
        elif len(common_cols) > 0:
            merge_cols = list(common_cols)
            print(f"Merging on: {merge_cols}")
            df = df.merge(df_promotions, on=merge_cols, how='left')
    except Exception as e:
        print(f"  (promotions table not accessible: {type(e).__name__})")
    
    print(f"✓ Final data: {len(df):,} rows, {len(df.columns)} columns")
    
    # Save data
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/raw/marketing_data.csv", index=False)
    df.to_parquet("data/raw/marketing_data.parquet", index=False)
    
    print(f"\n✓ Saved to data/raw/")
    print(f"  - marketing_data.csv")
    print(f"  - marketing_data.parquet")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst row:")
    print(df.iloc[0])
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
