# Marketing Mix Model (MMM)

A complete marketing mix modeling pipeline for analyzing the incremental impact of marketing channels on sales, with support for fixed effects, promotional controls, and Marginal ROAS optimization.

## What is This?

This repository implements a **statistical marketing mix model** that estimates:

1. **Channel Elasticity** - % change in revenue per % change in spend (e.g., "1% more Google spend → 0.023% more revenue")
2. **Marginal ROAS** - Revenue per additional dollar spent at current levels (e.g., "Next Facebook dollar → $1.82 revenue")
3. **Promotional Impact** - Revenue lift from promotions (holding spend constant)
4. **Channel Ranking** - Which channels are most profitable for incremental investment

**Key Features:**
- ✅ Log-log specification with fixed effects (geo & week)
- ✅ Carryover/adstock effects via lagged variables
- ✅ Seasonal controls (month, quarter, week patterns)
- ✅ Organic traffic isolated as control variable
- ✅ Promotional features extracted from text
- ✅ 312 engineered features from raw data
- ✅ OLS, Ridge, and Lasso model variants

**Data Grain:** Weekly × Geographic region (6,200 observations)

---

## Quick Start (5 minutes)

### 1. Run EDA Report
```bash
python eda_summary.py
```
Outputs: Data quality, channel spend, revenue metrics, promotional analysis

### 2. Run Full Analysis Pipeline
```bash
python run_mmm_analysis.py
```
Outputs: 
- Model coefficients (elasticities)
- Marginal ROAS by channel
- Promotional impact
- Model performance (R², RMSE, MAE)

### 3. Interactive Modeling (Jupyter)
```bash
jupyter notebook notebooks/03_mmm_modeling.ipynb
```
For step-by-step analysis and customization

---

## Installation

### Requirements
- Python 3.8+
- conda (recommended) or pip

### Setup Environment
```bash
# Using conda (recommended)
conda create -n marketing-mix-model python=3.10
conda activate marketing-mix-model
pip install -r requirements.txt

# Or activate existing environment
conda activate marketing-mix-model
```

### Package Dependencies
```
pandas==2.3.3
numpy==1.26.4
scikit-learn==1.3.2
google-cloud-bigquery==3.40.0
google-cloud-storage==2.13.0
pyyaml==6.0.1
```

---

## Data Pipeline

### Step 1: Data Acquisition
```bash
# Download fresh data from GCP BigQuery
python download_from_rekrutacja20.py
```
Saves to: `data/raw/marketing_data.csv`

**Raw Data Format:**
- **Rows:** 6,200 weeks × 40 geographic regions
- **Columns:** 20 features
  - `week` - Week date
  - `geo` - Geographic ID (Geo0-Geo39)
  - `cost_*` - Spend per channel (5 channels)
  - `impression_*` - Reach per channel
  - `revenue` - Target variable (total sales)
  - `conversions` - Total transactions
  - `promo_description` - Text description of active promotions
  - `population` - Region population
  - `market_share` - Regional market share

### Step 2: Exploratory Analysis
```bash
python eda_summary.py
```
Analyzes:
- Data quality & completeness
- Channel spend distribution
- Promotional patterns
- Geographic variation

### Step 3: Preprocessing & Feature Engineering
```python
import sys; sys.path.insert(0, 'src')
from preprocessing import preprocess_marketing_data
import pandas as pd

df = pd.read_csv('data/raw/marketing_data.csv')
df_processed, features = preprocess_marketing_data(df)
df_processed.to_csv('data/processed/marketing_data_processed.csv', index=False)
```

**Preprocessing Creates 312 Features:**

| Category | Count | Details |
|----------|-------|---------|
| **Spend** | 10 | Raw (5) + Log-transformed (5) |
| **Lagged Spend** | 11 | Lag 1 & 2 for all 5 channels |
| **Impressions** | 6 | Channel reach metrics |
| **CPM** | 5 | Cost per mille (efficiency) |
| **Geo Fixed Effects** | 39 | Market baseline (one per region) |
| **Week Fixed Effects** | 154 | Temporal baseline (one per week) |
| **Seasonality** | 65 | Month (11) + Quarter (3) + Week (51) |
| **Promotions** | 7 | Type flags + intensity |
| **Organic Traffic** | 2 | log_organic + lag1 |
| **Controls** | 2 | Population, market_share |

### Step 4: Model Fitting
```python
from mmm_model import MMModel

X = df_processed[feature_list].fillna(0)
y = df_processed['revenue']

model = MMModel(model_type='ols')  # or 'ridge', 'lasso'
model.fit(X, y, standardize=True)
metrics = model.evaluate(X, y)
```

### Step 5: Business Metrics
```python
# Marginal ROAS
mroas = model.calculate_marginal_roas(X, y, spend_features, raw_spend)

# Elasticity
coefs = model.get_coefficients()

# Contribution
contrib = model.calculate_contribution(X, y)
```

---

## Pipeline Configuration

### Preprocessing Config
Located in: `src/preprocessing.py::_default_config()`

```python
config = {
    'fill_tv_impressions': True,      # Fill 5.3% missing TV impressions
    'remove_outdoor2': True,           # Remove 100% missing channel
    'log_transform': True,             # Create log features
    'handle_promotions': True,         # Extract promo features
    'create_lags': True,               # Create lag 1, 2 variables
    'lag_periods': [1, 2],            # Which lags to create
    'create_fixed_effects': True,      # Create geo & week FE
    'create_seasonality': True,        # Create seasonal controls
}

df_processed, features = preprocess_marketing_data(df, config)
```

### Model Config
```python
# OLS (Ordinary Least Squares) - Most interpretable
model = MMModel(model_type='ols')
model.fit(X, y)

# Ridge - Add L2 regularization for stability
model = MMModel(model_type='ridge', alpha=1.0)
model.fit(X, y)

# Lasso - L1 regularization (feature selection)
model = MMModel(model_type='lasso', alpha=0.01)
model.fit(X, y)
```

---

## Feature Engineering Details

### 1. Media Spend Features
```python
# Raw spend per channel
'cost_tiktok', 'cost_tv', 'cost_outdoor1', 'cost_google', 'cost_facebook'

# Log-transformed (elasticity-friendly)
'cost_tiktok_log', 'cost_tv_log', ...

# Lagged (carryover effects)
'cost_tiktok_lag1', 'cost_tiktok_lag2', ...
```

### 2. Promotional Features (Regex-based)
```python
'free_shipping_promo'    # Pattern: "darmowa dostawa", "free shipping"
'discount_percent_promo'  # Pattern: "rabat", "-15%", "taniej"
'free_item_promo'         # Pattern: "gratis", "bezpłatnie"
'bundle_multibuy_promo'   # Pattern: "druga sztuka", "kup 2"
'min_purchase_promo'      # Pattern: "minimum", "zamówień powyżej"
'discount_percent'        # Extracted: 15 from "-15%" text
'promo_intensity'         # Count of active promos per week-geo
```

### 3. Fixed Effects
```python
# Geo fixed effects (one-hot encoded, drop_first=True)
'geo_fe_Geo1', 'geo_fe_Geo2', ..., 'geo_fe_Geo39'  # 39 dummies

# Week fixed effects (one-hot encoded, drop_first=True)
'week_fe_2022-01-03', 'week_fe_2022-01-10', ...    # 154 dummies
```

### 4. Seasonality Controls
```python
# Month of year (drop Nov to avoid collinearity)
'month_1', 'month_2', ..., 'month_11'  # 11 dummies

# Quarter (drop Q4)
'quarter_1', 'quarter_2', 'quarter_3'  # 3 dummies

# Week of year (drop week 52)
'week_of_year_1', 'week_of_year_2', ..., 'week_of_year_51'  # 51 dummies
```

---

## Model Specification

### Log-Log Specification (Elasticity)

$$\log(\text{revenue}_{it}) = \alpha_i + \gamma_t + \sum_k \beta_k \log(\text{spend}_{k,it} + 1) + \theta \log(\text{organic}_{it} + 1) + \delta \cdot \text{promo}_{it} + \text{controls} + \epsilon_{it}$$

**Where:**
- $\alpha_i$ = 39 geo fixed effects
- $\gamma_t$ = 154 week fixed effects
- $\beta_k$ = **Elasticity** of channel k (% revenue change per % spend change)
- $\theta$ = Organic traffic elasticity
- $\delta$ = Promotional impact (revenue lift)
- controls = Population, market_share, seasonality, lagged spend

**Interpretation:**
- β_google = 0.025 → 10% more Google spend → 0.25% more revenue
- β_facebook = 0.065 → 10% more Facebook spend → 0.65% more revenue

---

## Business Metrics

### 1. Elasticity
```python
Elasticity = β_k × (mean_input / mean_output)
```
**Interpretation:** % change in revenue per % change in spend
- Example: Google elasticity = 0.025 → inelastic (less sensitive)
- Example: Facebook elasticity = 0.065 → more elastic

### 2. Marginal ROAS
```python
MROAS_k = β_k × (Total_Revenue / Total_Spend_k)
```
**Interpretation:** Additional revenue per $1 incremental spend
- MROAS > 1.0 → Profitable channel (invest more)
- MROAS < 1.0 → Loss-making at current spend (reduce or optimize)

### 3. Channel Contribution
```python
Contribution = (β_k × mean_spend) / mean_revenue × 100%
```
**Interpretation:** % of total revenue attributed to each channel

### 4. Promotional Impact
Coefficients on:
- `free_shipping_promo` - Revenue lift from free shipping offers
- `discount_percent_promo` - Revenue from discount promotions
- `promo_intensity` - Diminishing returns from high promo frequency

---

## Usage Examples

### Complete Analysis Pipeline
```python
import sys; sys.path.insert(0, 'src')
import pandas as pd
from preprocessing import preprocess_marketing_data
from mmm_model import MMModel

# 1. Load and preprocess
df = pd.read_csv('data/raw/marketing_data.csv')
df_processed, features = preprocess_marketing_data(df)

# 2. Build feature set (balanced model)
feature_list = (
    features['spend_features_log'] +      # 5 log spend
    features['control_vars'] +            # 4 controls
    features['promotional_features'] +    # 7 promo
    features['geo_fixed_effects'] +       # 39 geo FE
    features['seasonal_features']         # 65 seasonality
)

X = df_processed[feature_list].fillna(0)
y = df_processed['revenue']

# 3. Fit model
model = MMModel(model_type='ols')
model.fit(X, y, standardize=True)
metrics = model.evaluate(X, y)

print(f"R² = {metrics['R²']:.4f}")

# 4. Get results
mroas = model.calculate_marginal_roas(X, y, features['spend_features_log'])
print(mroas.sort_values('mroas', ascending=False))
```

### Model Comparison
```python
# Compare OLS vs Ridge vs Lasso
models = {}
for model_type in ['ols', 'ridge', 'lasso']:
    m = MMModel(model_type=model_type, alpha=1.0)
    m.fit(X, y)
    metrics = m.evaluate(X, y)
    models[model_type] = metrics
    print(f"{model_type}: R² = {metrics['R²']:.4f}")
```

### Scenario Analysis (Budget Allocation)
```python
import numpy as np

# Current spend levels
current_spend = df[['cost_google', 'cost_facebook', 'cost_tiktok']].sum()

# Scenario: +20% Google spend
scenario_spend = current_spend.copy()
scenario_spend['cost_google'] *= 1.20

# Estimated revenue impact
coefs = model.get_coefficients()
revenue_increase = coefs['cost_google_log'] * np.log(1.20)
print(f"Expected revenue increase: {revenue_increase * 100:.2f}%")
```

---

## Directory Structure

```
marketing-mix-model/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
│
├── data/
│   ├── raw/
│   │   └── marketing_data.csv            # Raw data (6,200 rows × 20 cols)
│   └── processed/
│       └── marketing_data_processed.csv   # Engineered features (6,200 × 312)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                  # Feature engineering pipeline
│   ├── mmm_model.py                      # Model estimation & metrics
│   ├── data_loader.py                    # GCP/local data loading
│   └── steps/                            # Orchestration steps
│       ├── data_import.py
│       ├── data_export.py
│       ├── model_training.py
│       └── model_evaluation.py
│
├── notebooks/
│   ├── 01_eda_and_gcp_setup.ipynb       # Exploratory analysis
│   └── 03_mmm_modeling.ipynb             # Interactive modeling
│
├── examples/
│   └── 01_data_preprocessing_example.py  # Usage example
│
├── config/
│   ├── pipeline_config.yaml              # Pipeline settings
│   ├── example_configs.yaml              # Config examples
│   └── gcp_credentials.json              # GCP auth (gitignored)
│
├── models/
│   └── results/                          # Model outputs
│
└── logs/                                 # Pipeline logs
```

---

## Configuration

### Preprocessing Configuration
Located in: `src/preprocessing.py`

```python
{
    'fill_tv_impressions': True,      # Fill 5.3% missing TV impressions
    'remove_outdoor2': True,           # Remove 100% missing outdoor2 channel
    'log_transform': True,             # Create log-transformed features
    'handle_promotions': True,         # Extract promotional features
    'create_lags': True,               # Create lag 1-2 variables
    'lag_periods': [1, 2],            # Which lag periods to create
    'create_fixed_effects': True,      # Create geo & week fixed effects
    'create_seasonality': True,        # Create seasonal dummy variables
}
```

### Model Configuration
```python
# Model type selection
'ols'    → Ordinary Least Squares (interpretable, no regularization)
'ridge'  → Ridge regression (L2 penalty for stability)
'lasso'  → Lasso regression (L1 penalty for feature selection)

# Parameters
alpha    → Regularization strength (0.01-100 for ridge/lasso)
standardize → Normalize features before fitting (recommended: True)
```

---

## Data Quality

### Completeness
- **Raw Data:** 98.9% complete
- **Processed Data:** 99.9% complete

### Missing Values
| Column | Missing | Handling |
|--------|---------|----------|
| impression_tv | 5.3% | Median fill |
| impression_outdoor2 | 100% | Channel removed |
| promo_description | 46.8% | Expected |
| All others | <1% | Minimal |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "ModuleNotFoundError: pandas" | `pip install -r requirements.txt` |
| "Data file not found" | `python download_from_rekrutacja20.py` |
| "ImportError: preprocessing" | Verify `sys.path.insert(0, 'src')` |
| Negative R² | Reduce feature count (too many features for sample size) |
| Negative elasticity | May indicate weak relationship with revenue |

---

## Performance Benchmarks

- **Preprocessing:** ~2 seconds (312 features)
- **Model fitting (OLS):** <1 second
- **Full analysis:** ~5 seconds

---

## Advanced Features

### Adstock Transformation
```python
from mmm_model import AdstockTransformer

df_adstocked = AdstockTransformer.apply_adstock_to_channels(
    df,
    spend_columns=['cost_google', 'cost_facebook'],
    decay_rates={'cost_google': 0.5, 'cost_facebook': 0.7}
)
```

### Custom Configuration
```python
config = {
    'create_lags': True,
    'lag_periods': [1, 2, 3],
    'create_seasonality': False,
}
df_processed, features = preprocess_marketing_data(df, config)
```

---

## License

Proprietary - Internal Use Only

---

**Last Updated:** January 30, 2026  
**Status:** ✅ Production Ready