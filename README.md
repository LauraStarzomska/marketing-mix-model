# marketing-mix-model
# Marketing Mix Model (MMM) Pipeline

A modular, production-ready Marketing Mix Modeling pipeline with GCP integration for analyzing marketing channel effectiveness and ROI.

## Overview

This project implements a complete MMM pipeline that:
- Imports data from Google Cloud BigQuery or local CSV files
- Preprocesses marketing data with feature engineering (adstock, saturation)
- Trains regression models (Ridge, Lasso, OLS) to analyze channel contributions
- Evaluates model performance and calculates marketing elasticity
- Exports results to local storage and/or Google Cloud Storage

## Project Structure

```
marketing-mix-model/
├── config/
│   ├── pipeline_config.yaml       # Main pipeline configuration
│   └── example_configs.yaml       # Example configurations
├── src/
│   ├── mmm_model.py              # Core MMM model class
│   ├── preprocessing.py           # Data preprocessing & feature engineering
│   └── steps/                     # Modular pipeline steps
│       ├── data_import.py        # GCP/local data import
│       ├── model_training.py     # Model training
│       ├── model_evaluation.py   # Metrics & elasticity calculation
│       └── data_export.py        # Local/GCS export
├── notebooks/
│   └── 01_eda_and_gcp_setup.ipynb  # EDA and setup guide
├── examples/
│   ├── 01_data_preprocessing_example.py
│   └── README.md
├── pipeline_orchestrator.py       # Main pipeline orchestrator
├── deploy_to_cloudrun.sh         # GCP Cloud Run deployment script
├── Dockerfile                     # Container specification
├── gcp_cloud_workflow.yaml       # Cloud Workflow definition
├── requirements.txt               # Python dependencies
└── README.md                      # This file

```

## Quick Start

### Prerequisites
- Python 3.11+
- gcloud CLI (for GCP integration)
- Docker (for Cloud Run deployment)
- GCP project with BigQuery and Cloud Storage enabled

### Installation

1. Clone the repository:
```bash
git clone https://github.com/LauraStarzomska/marketing-mix-model.git
cd marketing-mix-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure GCP authentication:
```bash
gcloud auth login
gcloud config set project alterdata-rekrutacja-20
gcloud auth application-default login
```

### Running Locally

1. Update configuration in `config/pipeline_config.yaml`:
```yaml
data:
  source: 'gcp'  # or 'local'
  project: 'alterdata-rekrutacja-20'
  dataset: 'marketing_dataset'
  table: 'campaigns'

model:
  type: 'ridge'
  alpha: 1.0

output:
  local:
    enabled: true
    dir: 'models/results'
  gcs:
    enabled: false  # Set to true for GCS export
    bucket: 'your-bucket-name'
```

2. Run the pipeline:
```bash
python pipeline_orchestrator.py
```

## Deployment to GCP Cloud Run

### Automated Deployment

Run the deployment script:
```bash
./deploy_to_cloudrun.sh
```

This will:
- Build Docker image
- Push to Artifact Registry
- Deploy to Cloud Run
- Configure service with 2GB memory, 2 CPU, 600s timeout

### Manual Deployment

```bash
# Build image
docker build -t mmm-pipeline .

# Tag for Artifact Registry
docker tag mmm-pipeline us-central1-docker.pkg.dev/alterdata-rekrutacja-20/cloud-run-source/mmm-pipeline

# Push to registry
docker push us-central1-docker.pkg.dev/alterdata-rekrutacja-20/cloud-run-source/mmm-pipeline

# Deploy to Cloud Run
gcloud run deploy mmm-pipeline \
  --image us-central1-docker.pkg.dev/alterdata-rekrutacja-20/cloud-run-source/mmm-pipeline \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 600s
```

## Configuration

### Pipeline Configuration (`config/pipeline_config.yaml`)

```yaml
steps:
  data_import:
    enabled: true
  model_training:
    enabled: true
  model_evaluation:
    enabled: true
  data_export:
    enabled: true

data:
  source: 'gcp'              # 'gcp' or 'local'
  project: 'alterdata-rekrutacja-20'
  dataset: 'marketing_dataset'
  table: 'campaigns'
  path: 'data/marketing_data.csv'  # For local source

model:
  type: 'ridge'              # 'ridge', 'lasso', or 'ols'
  alpha: 1.0                 # Regularization parameter

preprocessing:
  adstock_rate: 0.5
  saturation_factor: 0.0001

output:
  local:
    enabled: true
    dir: 'models/results'
  gcs:
    enabled: false
    project: 'alterdata-rekrutacja-20'
    bucket: 'your-bucket-name'
    path: 'mmm/results'
```

## Features

### Data Processing
- **Adstock transformation**: Models carryover effects of marketing
- **Saturation modeling**: Captures diminishing returns
- **Automated type conversion**: Handles BigQuery string types

### Model Training
- **Multiple algorithms**: Ridge, Lasso, OLS regression
- **Hyperparameter tuning**: Configurable regularization
- **Feature engineering**: Automated preprocessing pipeline

### Evaluation & Insights
- **Model metrics**: R², RMSE, MAE, MSE
- **Marketing elasticity**: Channel-level sensitivity analysis
- **Revenue contribution**: Percentage attribution by channel
- **Export formats**: JSON, CSV, pickle

### Cloud Integration
- **BigQuery data import**: Direct SQL query execution
- **Cloud Storage export**: Automated result uploads
- **Cloud Run deployment**: Containerized production service
- **Cloud Logging**: Structured log output

## Usage Examples

### Example 1: Local CSV Processing
```yaml
data:
  source: 'local'
  path: 'data/marketing_data.csv'
```

### Example 2: GCP with GCS Export
```yaml
data:
  source: 'gcp'
  project: 'alterdata-rekrutacja-20'
  
output:
  gcs:
    enabled: true
    bucket: 'mmm-results-bucket'
```

### Example 3: Lasso Model with Custom Parameters
```yaml
model:
  type: 'lasso'
  alpha: 0.5

preprocessing:
  adstock_rate: 0.7
  saturation_factor: 0.00015
```

## Development

### Project Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements.txt
```

### Running Tests
```bash
# Test pipeline locally
python pipeline_orchestrator.py

# Test Docker build
docker build -t mmm-pipeline .
docker run mmm-pipeline
```

## Documentation

- [HOW_TO_RUN.md](HOW_TO_RUN.md) - Detailed usage instructions
- [GCP_DEPLOYMENT.md](GCP_DEPLOYMENT.md) - Cloud deployment guide
- [examples/README.md](examples/README.md) - Code examples
- [notebooks/01_eda_and_gcp_setup.ipynb](notebooks/01_eda_and_gcp_setup.ipynb) - EDA notebook

## Architecture

The pipeline follows a modular step-based architecture:

1. **Data Import** (`data_import.py`)
   - Loads from BigQuery or local CSV
   - Handles type conversions
   - Validates data structure

2. **Model Training** (`model_training.py`)
   - Applies preprocessing transformations
   - Trains regression model
   - Passes artifacts to next step

3. **Model Evaluation** (`model_evaluation.py`)
   - Calculates performance metrics
   - Computes channel elasticity
   - Determines revenue contribution

4. **Data Export** (`data_export.py`)
   - Saves to local filesystem
   - Uploads to Cloud Storage
   - Supports multiple formats

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes and commit: `git commit -m "Add feature"`
3. Push to branch: `git push origin feature/your-feature`
4. Create Pull Request

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue.
