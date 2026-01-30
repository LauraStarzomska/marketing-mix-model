#!/usr/bin/env python
"""
Quick start: MMM with fixed effects and MROAS calculation

Run this script to:
1. Preprocess data with fixed effects, lags, seasonality
2. Fit OLS model with promotional controls
3. Calculate Marginal ROAS for each channel
4. Generate elasticity analysis
"""

import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from preprocessing import preprocess_marketing_data
from mmm_model import MMModel

print("=" * 80)
print("MARKETING MIX MODEL - FIXED EFFECTS + MARGINAL ROAS")
print("=" * 80)

# 1. LOAD AND PREPROCESS DATA
print("\n[1/4] Loading and preprocessing data...\n")
df_raw = pd.read_csv('data/raw/marketing_data.csv')
df_processed, features = preprocess_marketing_data(df_raw)

# 2. BUILD FEATURE SET
print("\n[2/4] Building model feature set...\n")

# Core model features (recommended balanced approach)
spend_features = features['spend_features_log']
control_features = features['control_vars']
promo_features = features['promotional_features']

# Optional: Add fixed effects (commented for computational efficiency)
# Uncomment to include full fixed effects (will increase variance)
geo_fe = features['geo_fixed_effects'][:10]  # Subset of geo FE
seasonal_features = features['seasonal_features'][:20]  # Subset of seasonality

# Build full feature matrix
feature_list = spend_features + control_features + promo_features + geo_fe + seasonal_features
X = df_processed[feature_list].fillna(0)
y = df_processed['revenue']

print(f"Target variable: revenue")
print(f"Features used: {len(feature_list)}")
print(f"  - Spend (log): {len(spend_features)}")
print(f"  - Organic controls: {len(control_features)}")
print(f"  - Promotions: {len(promo_features)}")
print(f"  - Geo FE (sample): {len(geo_fe)}")
print(f"  - Seasonality: {len(seasonal_features)}")
print(f"Sample size: {len(X)}")

# 3. FIT MODEL
print("\n[3/4] Fitting OLS model...\n")
model = MMModel(model_type='ols')
model.fit(X, y, standardize=True)
metrics = model.evaluate(X, y, standardize=True)

print(f"\nModel Performance:")
print(f"  R²:   {metrics['R²']:.4f}")
print(f"  RMSE: ${metrics['RMSE']:>12,.0f}")
print(f"  MAE:  ${metrics['MAE']:>12,.0f}")

# 4. CALCULATE BUSINESS METRICS
print("\n[4/4] Computing business metrics...\n")

# Get model coefficients
coefs = model.get_coefficients()
print("\nChannel Coefficients (Elasticities in log-log model):")
for spend_feat in spend_features:
    channel = spend_feat.replace('cost_', '').replace('_log', '')
    beta = coefs.get(spend_feat, 0)
    print(f"  {channel:12} β = {beta:>8.6f}")

# Calculate Marginal ROAS
print("\nCalculating Marginal ROAS...")
mroas_df = model.calculate_marginal_roas(
    X, 
    y,
    spend_features=spend_features,
    raw_spend=df_raw[['cost_tiktok', 'cost_tv', 'cost_outdoor1', 'cost_google', 'cost_facebook']]
)

print("\n" + "=" * 80)
print("MARGINAL ROAS BY CHANNEL (Revenue per $1 incremental spend)")
print("=" * 80)
mroas_ranked = mroas_df.sort_values('mroas', ascending=False)
for channel, row in mroas_ranked.iterrows():
    channel_name = channel.replace('cost_', '').replace('_log', '')
    print(f"  {channel_name:12} MROAS = ${row['mroas']:>7.2f}  (β = {row['coefficient']:>8.6f})")

# Get elasticities
print("\n" + "=" * 80)
print("ELASTICITY BY CHANNEL (% change in revenue per % change in spend)")
print("=" * 80)
for spend_feat in spend_features:
    channel = spend_feat.replace('cost_', '').replace('_log', '')
    if spend_feat in coefs.index:
        beta = coefs[spend_feat]
        elasticity = beta * (X[spend_feat].mean() / y.mean())
        print(f"  {channel:12} elasticity = {elasticity:>7.4f}")

# Promotional impact
print("\n" + "=" * 80)
print("PROMOTIONAL IMPACT")
print("=" * 80)
print("\nPromotion Coefficients:")
for promo_feat in promo_features:
    if promo_feat in coefs.index:
        coef = coefs[promo_feat]
        impact = f"${coef * y.mean():,.0f}" if promo_feat == 'has_promotion' else f"{coef:.4f}"
        print(f"  {promo_feat:30} = {impact:>15}")

# Summary recommendations
print("\n" + "=" * 80)
print("INTERPRETATION GUIDE")
print("=" * 80)
print("""
1. ELASTICITY: For 1% increase in channel spend, revenue changes by elasticity%
   - Example: Google elasticity = 0.0234 means 10% more spend → 0.234% more revenue

2. MARGINAL ROAS: Additional revenue per extra $1 spent (at current spend levels)
   - Example: Facebook MROAS = $1.82 means 1 more dollar → $1.82 revenue
   - Action: Invest in channels with MROAS > 1.0

3. PROMOTION IMPACT: Revenue effect of running a promotion (holding spend constant)
   - Positive = promotions drive incremental revenue
   - Use to prioritize promo intensity vs. discount depth

4. FIXED EFFECTS:
   - Geo FE: Market-specific baselines (account for regional differences)
   - Seasonality: Captures month/quarter patterns

5. ORGANIC TRAFFIC: Isolates paid media impact from organic/SEO traffic
   - Controls for external demand shocks
""")

print("\n" + "=" * 80)
print("✓ Analysis Complete!")
print("=" * 80)
print(f"""
Next Steps:
1. Review MROAS rankings for budget allocation decisions
2. Check elasticity signs (should be non-negative for all channels)
3. Validate promotion impact with marketing team feedback
4. Consider lagged effects (is week t-1 spend impacting week t revenue?)
5. Run sensitivity analysis: "What if we increased Google spend by 20%?"

For more details, see:
  - MMM_COPILOT_IMPLEMENTATION.md (full specification)
  - notebooks/03_mmm_modeling.ipynb (interactive analysis)
""")
