"""
Quick EDA Summary Report Generator

Generates a fast overview of marketing data without opening Jupyter.
Includes analysis of promotional descriptions for feature engineering.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessing import MarketingDataPreprocessor


def print_header(title):
    """Print formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_section(title):
    """Print formatted subsection"""
    print(f"\n{title}")
    print("-" * 70)


def analyze_promotional_data(df):
    """Analyze promotional descriptions for feature engineering opportunities"""
    print_section("📢 PROMOTIONAL DATA ANALYSIS")
    
    if 'promo_description' not in df.columns:
        print("  ⚠ No promotional data found")
        return
    
    # Basic statistics
    total_rows = len(df)
    promo_rows = df['promo_description'].notna().sum()
    promo_percent = (promo_rows / total_rows) * 100
    
    print(f"  • Total rows: {total_rows:,}")
    print(f"  • Rows with promotions: {promo_rows:,} ({promo_percent:.1f}%)")
    print(f"  • Rows without promotions: {total_rows - promo_rows:,} ({100-promo_percent:.1f}%)")
    
    # Unique promotions
    unique_promos = df['promo_description'].nunique()
    print(f"  • Unique promotion types: {unique_promos}")
    
    # Promotion keywords analysis
    print_section("  🔍 Promotion Keywords (Frequency)")
    
    keywords = {
        'Free Shipping': ['darmowa dostawa', 'bezpłatna wysyłka', 'free shipping'],
        'Discount %': ['rabat', 'discount', '%', 'taniej'],
        'Free Item': ['gratis', 'bezpłatnie', 'free'],
        'Bundle/Multi-buy': ['druga sztuka', 'kupujesz', 'przy zakupie'],
        'Min Purchase': ['zamówień powyżej', 'minimum', 'od'],
        'Time Limited': ['ograniczona', 'limited', 'koniec'],
    }
    
    promo_text = df['promo_description'].dropna().str.lower()
    
    keyword_counts = {}
    for category, patterns in keywords.items():
        count = promo_text.str.contains('|'.join(patterns), regex=True, case=False).sum()
        if count > 0:
            pct = (count / promo_rows) * 100
            keyword_counts[category] = (count, pct)
            print(f"    • {category}: {count:,} ({pct:.1f}%)")
    
    # Sample promotions
    print_section("  📋 Sample Promotions (First 5)")
    for i, promo in enumerate(df['promo_description'].dropna().head(5), 1):
        print(f"    {i}. {promo[:65]}..." if len(promo) > 65 else f"    {i}. {promo}")
    
    # Geographic promotion distribution
    print_section("  🗺️ Promotion Distribution by Geography")
    geo_promo = df.groupby('geo')['promo_description'].apply(
        lambda x: (x.notna().sum(), len(x))
    ).apply(lambda x: f"{x[0]:>5} / {x[1]:<5} ({100*x[0]/x[1]:>5.1f}%)")
    
    for geo, stats in geo_promo.items():
        print(f"    {geo}: {stats}")
    
    # Feature engineering suggestions
    print_section("  💡 Feature Engineering Suggestions")
    print("  1. Binary indicator: has_promotion (0/1)")
    print(f"     → Create: df['has_promotion'] = df['promo_description'].notna().astype(int)")
    print(f"\n  2. Promotion types (One-Hot Encoding):")
    for keyword in keyword_counts.keys():
        print(f"     → {keyword.lower()}_promo")
    print(f"\n  3. Promotion intensity:")
    print(f"     → Count of active promotions per week/geo")
    print(f"     → Days since last promotion per channel")
    print(f"\n  4. NLP features (optional):")
    print(f"     → Sentiment of promotional text")
    print(f"     → Discount magnitude extraction (%)")
    print(f"     → Free shipping indicator")


def analyze_data_quality(df):
    """Analyze data quality and completeness"""
    print_section("📊 DATA QUALITY REPORT")
    
    print(f"  Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Date Range: {df['week'].min()} to {df['week'].max()}")
    print(f"  Geographic Coverage: {df['geo'].nunique()} regions")
    
    # Missing values
    print_section("  ⚠️ Missing Values")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_summary = pd.DataFrame({
        'Missing Count': missing[missing > 0],
        'Missing %': missing_pct[missing > 0]
    }).sort_values('Missing Count', ascending=False)
    
    if len(missing_summary) > 0:
        for col, row in missing_summary.iterrows():
            print(f"    • {col}: {int(row['Missing Count']):,} ({row['Missing %']:.1f}%)")
    else:
        print("    ✓ No missing values!")


def analyze_numeric_features(df):
    """Analyze numeric features and channels"""
    print_section("💰 CHANNEL & FINANCIAL METRICS")
    
    # Identify channels
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cost_cols = [c for c in numeric_cols if 'cost_' in c]
    impression_cols = [c for c in numeric_cols if 'impression_' in c]
    
    channels = {c.replace('cost_', ''): c for c in cost_cols}
    
    print(f"  Active Channels: {', '.join(channels.keys())}")
    
    # Channel spend summary
    print_section("  💵 Spend Summary (Total, Mean, Min, Max)")
    spend_data = []
    for channel in sorted(channels.keys()):
        cost_col = f'cost_{channel}'
        if cost_col in df.columns:
            total = df[cost_col].sum()
            mean = df[cost_col].mean()
            min_spend = df[cost_col].min()
            max_spend = df[cost_col].max()
            
            spend_data.append({
                'Channel': channel,
                'Total Spend': f"${total:,.0f}",
                'Avg/Week': f"${mean:,.0f}",
                'Min': f"${min_spend:,.0f}",
                'Max': f"${max_spend:,.0f}"
            })
    
    if spend_data:
        for row in sorted(spend_data, key=lambda x: float(x['Total Spend'].replace('$', '').replace(',', '')), reverse=True):
            print(f"    {row['Channel']:12} • Total: {row['Total Spend']:>12} | Avg: {row['Avg/Week']:>12} | Range: [{row['Min']:>12}, {row['Max']:>12}]")
    
    # Impressions
    print_section("  📈 Impressions Summary (Total, Mean)")
    for channel in sorted(channels.keys()):
        imp_col = f'impression_{channel}'
        if imp_col in df.columns:
            total_imp = df[imp_col].sum()
            mean_imp = df[imp_col].mean()
            print(f"    {channel:12} • Total: {total_imp:>15,.0f} | Avg: {mean_imp:>12,.0f}/week")
    
    # Revenue
    print_section("  🎯 Revenue Metrics")
    if 'revenue' in df.columns:
        total_revenue = df['revenue'].sum()
        mean_revenue = df['revenue'].mean()
        print(f"    • Total Revenue: ${total_revenue:,.0f}")
        print(f"    • Weekly Revenue (avg): ${mean_revenue:,.0f}")
        print(f"    • Weekly Revenue (min): ${df['revenue'].min():,.0f}")
        print(f"    • Weekly Revenue (max): ${df['revenue'].max():,.0f}")
    
    if 'conversions' in df.columns:
        print(f"    • Total Conversions: {df['conversions'].sum():,.0f}")
        print(f"    • Avg Conversions/Week: {df['conversions'].mean():,.0f}")


def main():
    """Main execution"""
    print_header("MARKETING MIX MODEL - QUICK EDA REPORT")
    
    data_path = Path(__file__).parent / 'data' / 'raw' / 'marketing_data.csv'
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("\nTo download data, run:")
        print("  python download_from_rekrutacja20.py")
        return
    
    print(f"📂 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded successfully\n")
    
    # Run analyses
    analyze_data_quality(df)
    analyze_numeric_features(df)
    analyze_promotional_data(df)
    
    # Summary statistics
    print_section("📋 SUMMARY STATISTICS (All Numeric Columns)")
    print(df.describe().round(2).to_string())
    
    print_header("✅ REPORT COMPLETE")
    print(f"\nNext Steps:")
    print(f"  1. Review promotional data features above")
    print(f"  2. Preprocess data: python -c \"import sys; sys.path.insert(0, 'src'); from preprocessing import preprocess_marketing_data; import pandas as pd; df=pd.read_csv('data/raw/marketing_data.csv'); df_p, features = preprocess_marketing_data(df); df_p.to_csv('data/processed/marketing_data_processed.csv', index=False); print('✓ Data preprocessed')\"")
    print(f"  3. Open modeling notebook: jupyter notebook notebooks/03_mmm_modeling.ipynb")
    print(f"  4. Consider adding promo features: 'has_promotion', 'free_shipping', 'discount_percent'")
    print()


if __name__ == '__main__':
    main()
