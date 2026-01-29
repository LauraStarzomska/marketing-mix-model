# How to Run the Marketing Mix Model

## Current State

You have **three ways** to run the MMM analysis:

### Option 1: Jupyter Notebooks (Recommended for Exploration)

#### A. Exploratory Data Analysis (EDA)
```bash
# Open in Jupyter
jupyter notebook notebooks/01_eda_and_gcp_setup.ipynb
```
This gives you:
- Data schema and quality overview
- Temporal trends visualization
- Channel spend analysis
- Geographic performance
- ROAS by channel

#### B. Marketing Mix Model Building (Under Development)
```bash
jupyter notebook notebooks/03_mmm_modeling.ipynb
```
Currently has skeleton cells for:
- Data loading with GCP fallback
- Data preparation
- Model building
- Elasticity calculation

**Status:** Needs execution to complete

---

### Option 2: Python Scripts (Production Pipeline)

#### A. Download Fresh Data
```bash
python download_from_rekrutacja20.py
```
Fetches latest data from GCP BigQuery and saves to `data/raw/`

#### B. Quick EDA Summary
```bash
python eda_summary.py
```
Prints quick statistics without opening Jupyter

#### C. Data Preprocessing
```bash
python -c "
import sys
sys.path.insert(0, 'src')
import pandas as pd
from preprocessing import preprocess_marketing_data

df = pd.read_csv('data/raw/marketing_data.csv')
df_processed, features = preprocess_marketing_data(df)
df_processed.to_csv('data/processed/marketing_data_processed.csv', index=False)
"
```
Creates processed data with:
- Missing values filled
- New features engineered (CPM, log transforms)
- Unused channels removed

#### D. Run Full Pipeline Example
```bash
python examples/01_data_preprocessing_example.py
```

---

### Option 3: Python API (For Integration)

```python
import sys
sys.path.insert(0, 'src')

import pandas as pd
from preprocessing import preprocess_marketing_data
from mmm_model import MMModel

# 1. Load data
df = pd.read_csv('data/raw/marketing_data.csv')

# 2. Preprocess
df_processed, features = preprocess_marketing_data(df)

# 3. Prepare for modeling
X = df_processed[features['spend_features']]
y = df_processed[features['target']].values.ravel()

# 4. Build model
model = MMModel(model_type='ridge', alpha=1.0)
model.fit(X, y)

# 5. Get results
metrics = model.get_metrics()
elasticity = model.calculate_elasticity(X, features['spend_features'])
contribution = model.calculate_contribution(X, features['spend_features'])

print(f"R² Score: {metrics['r2']:.4f}")
print(f"Elasticity:\n{elasticity}")
print(f"Contribution:\n{contribution}")
```

---

## Complete Workflow (Recommended)

For full analysis from raw data to model results:

### 1. Data Acquisition
```bash
# Download latest data
python download_from_rekrutacja20.py
```

### 2. Exploratory Analysis
```bash
# View data quality and trends
jupyter notebook notebooks/01_eda_and_gcp_setup.ipynb
# Run through all cells
```

### 3. Data Preprocessing
```bash
# Create processed dataset
python -c "
import sys; sys.path.insert(0, 'src')
import pandas as pd
from preprocessing import preprocess_marketing_data
df = pd.read_csv('data/raw/marketing_data.csv')
df_p, features = preprocess_marketing_data(df)
df_p.to_csv('data/processed/marketing_data_processed.csv', index=False)
"
```

### 4. Model Building
```bash
# Execute MMM modeling notebook (currently needs work)
jupyter notebook notebooks/03_mmm_modeling.ipynb
```

### 5. Results & Reporting
Check `reports/` directory for output visualizations

---

## Current Data Status

✅ **Available:**
- Raw data: `data/raw/marketing_data.csv` (6,200 rows)
- Processed data: `data/processed/marketing_data_processed.csv` (6,200 rows, 35 columns)

✅ **Modules Ready:**
- `src/data_loader.py` - Load from GCP or local
- `src/preprocessing.py` - Full preprocessing pipeline
- `src/mmm_model.py` - Core MMM models (Ridge, OLS, Lasso)

⏳ **In Progress:**
- `notebooks/03_mmm_modeling.ipynb` - Needs completion

---

## Features Available in Model

### Preprocessing creates:
- **5 Cost columns** (spend per channel)
- **6 Impression columns** (reach per channel) 
- **5 CPM features** (efficiency metrics)
- **11 Log-transformed features** (elasticity-friendly)
- **1 Promotion indicator** (marketing synergies)

### Model can calculate:
- **Elasticity** - % revenue change per % spend change
- **Contribution** - Revenue attributed to each channel
- **ROAS** - Return on ad spend by channel
- **Feature importance** - Which channels drive revenue

---

## Quick Start (5 minutes)

```bash
# 1. See what data we have
python eda_summary.py

# 2. Preprocess and save
python -c "
import sys; sys.path.insert(0, 'src')
import pandas as pd
from preprocessing import preprocess_marketing_data
df_raw = pd.read_csv('data/raw/marketing_data.csv')
df, features = preprocess_marketing_data(df_raw)
print('Processed data shape:', df.shape)
"

# 3. Open notebook for interactive analysis
jupyter notebook notebooks/03_mmm_modeling.ipynb
```

---

## Troubleshooting

**"Module not found" error:**
```bash
cd /path/to/project
python -c "import sys; sys.path.insert(0, 'src'); from preprocessing import preprocess_marketing_data"
```

**Data not found:**
- Check `data/raw/marketing_data.csv` exists
- If not: `python download_from_rekrutacja20.py`

**GCP auth issues:**
- Already configured via gcloud login
- Falls back to local CSV automatically

---

## Next Steps

1. **Complete 03_mmm_modeling.ipynb** - Run all cells to build and test model
2. **Create model comparison** - Compare OLS vs Ridge vs Lasso
3. **Scenario analysis** - "What if we increased Google spend by 20%?"
4. **Optimization** - Find optimal budget allocation across channels
