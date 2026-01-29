# Examples Directory

This directory contains example scripts demonstrating how to use the MMM pipeline and modules.

## Available Examples

### 01_data_preprocessing_example.py
Demonstrates the data preprocessing pipeline:
- Loading raw data
- Running preprocessing transformations
- Saving processed data
- Displaying feature sets for modeling

**Usage:**
```bash
cd .. && python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'src')

import pandas as pd
from preprocessing import preprocess_marketing_data

df = pd.read_csv('data/raw/marketing_data.csv')
df_processed, features = preprocess_marketing_data(df)
"
```

## Adding New Examples

When adding new examples:
1. Name files with a prefix number: `XX_description.py`
2. Include detailed comments explaining each step
3. Use the actual module imports from `src/`
4. Save outputs to `data/processed/` or `models/`
5. Update this README with usage instructions
