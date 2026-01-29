# MMM Pipeline - GCP Deployment Guide

## Overview

Your MMM pipeline can run in **three ways**:

### 1. Local Python Script (Development/Testing)
### 2. GCP Cloud Function (On-demand via API)
### 3. GCP Cloud Scheduler (Automated/Scheduled runs)

---

## Option 1: Run Locally (Recommended First)

### Quick Start
```bash
# Single command with defaults (local data, Ridge model)
python mmm_pipeline.py

# With GCP data
python mmm_pipeline.py --data-source gcp

# Custom configuration
python mmm_pipeline.py \
  --data-source local \
  --data-path data/raw/marketing_data.csv \
  --model-type ridge \
  --model-alpha 1.0 \
  --output-dir models/results/
```

### Output
- `models/results/mmm_results_YYYYMMDD_HHMMSS_results.json` - Full results
- `models/results/mmm_results_YYYYMMDD_HHMMSS_processed_data.csv` - Processed data
- `models/results/mmm_results_YYYYMMDD_HHMMSS_model.pkl` - Saved model object

### Python API
```python
from mmm_pipeline import MMMPipeline

config = {
    'data_source': 'local',
    'data_path': 'data/raw/marketing_data.csv',
    'model_type': 'ridge',
    'model_alpha': 1.0,
    'save_results': True,
    'output_dir': 'models/results/'
}

pipeline = MMMPipeline(config)
results = pipeline.run()

# Access results
print(f"R² Score: {results['model']['metrics']['r2_score']:.4f}")
print(f"Elasticity: {results['elasticity']}")
print(f"Contribution: {results['contribution']}")
```

---

## Option 2: Deploy to GCP Cloud Functions (On-Demand)

### Prerequisites
```bash
# Install gcloud CLI
brew install google-cloud-sdk  # macOS
# or: curl https://sdk.cloud.google.com | bash  # Linux

# Initialize gcloud
gcloud init
gcloud auth login
gcloud config set project alterdata-rekrutacja-20
```

### Deployment Steps

#### Step 1: Create requirements.txt for Cloud Function
```bash
cat > requirements-gcp.txt << 'EOF'
pandas==2.3.3
numpy==1.26.4
scikit-learn==1.3.2
google-cloud-bigquery==3.40.0
google-cloud-storage==2.13.0
functions-framework==3.4.0
EOF
```

#### Step 2: Deploy Function
```bash
# Deploy HTTP-triggered Cloud Function
gcloud functions deploy mmm_pipeline_function \
  --runtime python311 \
  --trigger-http \
  --allow-unauthenticated \
  --entry-point mmm_pipeline_function \
  --source . \
  --memory 2GB \
  --timeout 600s \
  --set-env-vars PROJECT_ID=alterdata-rekrutacja-20

# Note the function URL in the output
# Example: https://region-project.cloudfunctions.net/mmm_pipeline_function
```

#### Step 3: Call the Function
```bash
# Basic call (uses defaults)
curl -X POST https://region-project.cloudfunctions.net/mmm_pipeline_function

# With custom config
curl -X POST https://region-project.cloudfunctions.net/mmm_pipeline_function \
  -H "Content-Type: application/json" \
  -d '{
    "data_source": "gcp",
    "model_type": "ridge",
    "model_alpha": 1.0
  }'

# From Python
import requests
url = "https://region-project.cloudfunctions.net/mmm_pipeline_function"
response = requests.post(url, json={
    "data_source": "gcp",
    "model_type": "ridge"
})
print(response.json())
```

### Check Logs
```bash
gcloud functions logs read mmm_pipeline_function --limit=100
```

---

## Option 3: Schedule on GCP (Daily/Weekly/Custom)

### Setup Cloud Pub/Sub Topic
```bash
# Create topic
gcloud pubsub topics create mmm-pipeline-trigger

# Create subscription (optional, for testing)
gcloud pubsub subscriptions create mmm-trigger-sub \
  --topic=mmm-pipeline-trigger
```

### Deploy Scheduled Cloud Function
```bash
# Deploy with Pub/Sub trigger
gcloud functions deploy mmm_pipeline_scheduled \
  --runtime python311 \
  --trigger-topic mmm-pipeline-trigger \
  --entry-point mmm_pipeline_scheduled \
  --source . \
  --memory 2GB \
  --timeout 600s \
  --set-env-vars PROJECT_ID=alterdata-rekrutacja-20
```

### Create Cloud Scheduler Job (Daily at 2 AM)
```bash
gcloud scheduler jobs create pubsub mmm-pipeline-daily \
  --schedule="0 2 * * *" \
  --timezone="America/New_York" \
  --topic=mmm-pipeline-trigger \
  --message-body='{"data_source":"gcp","model_type":"ridge"}'

# Enable the job
gcloud scheduler jobs resume mmm-pipeline-daily

# View jobs
gcloud scheduler jobs list

# View job details
gcloud scheduler jobs describe mmm-pipeline-daily

# Trigger immediately (for testing)
gcloud scheduler jobs run mmm-pipeline-daily

# Check execution history
gcloud scheduler jobs describe mmm-pipeline-daily --log-type executions
```

### Change Schedule Examples
```bash
# Weekly (Mondays at 3 AM)
--schedule="0 3 * * 1"

# Every 6 hours
--schedule="0 */6 * * *"

# Every 30 minutes
--schedule="*/30 * * * *"

# First day of month at midnight
--schedule="0 0 1 * *"
```

---

## Option 4: Deploy as Cloud Run (More Flexible)

Cloud Run is better if you need more control or want a long-running service.

```bash
# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "mmm_pipeline.py", "--data-source", "gcp"]
EOF

# Build and push to Container Registry
gcloud builds submit --tag gcr.io/alterdata-rekrutacja-20/mmm-pipeline

# Deploy to Cloud Run
gcloud run deploy mmm-pipeline \
  --image gcr.io/alterdata-rekrutacja-20/mmm-pipeline \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --timeout 600s \
  --set-env-vars PROJECT_ID=alterdata-rekrutacja-20
```

---

## Storing Results in Cloud Storage

Modify pipeline to save to Google Cloud Storage:

```python
# In mmm_pipeline.py, modify _save_results():

from google.cloud import storage

def _save_results(self):
    """Save results to Cloud Storage"""
    if self.config['save_results']:
        bucket_name = 'your-bucket-name'
        prefix = self.config['output_prefix']
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Save JSON results
        blob = bucket.blob(f"mmm_results/{prefix}_results.json")
        blob.upload_from_string(
            json.dumps(self.results),
            content_type='application/json'
        )
        
        logger.info(f"Results saved to gs://{bucket_name}/mmm_results/{prefix}_results.json")
```

---

## Monitoring & Debugging

### View Logs
```bash
# Cloud Function logs
gcloud functions logs read mmm_pipeline_function --limit=50 --follow

# Cloud Run logs
gcloud run logs read mmm-pipeline --limit=50 --follow

# Cloud Scheduler execution logs
gcloud logging read "resource.type=cloud_scheduler_job" --limit=10
```

### Set Up Alerts
```bash
# Alert if function errors exceed threshold
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="MMM Pipeline Errors" \
  --condition-display-name="High Error Rate" \
  --condition-threshold-value=5 \
  --condition-threshold-duration=300s \
  --condition-threshold-comparison=COMPARISON_GT
```

### Cost Considerations
- **Cloud Functions:** $0.40 per 1M invocations (free tier: 2M/month)
- **Cloud Run:** $0.00001667 per vCPU-second (free tier: 180,000 vCPU-seconds/month)
- **BigQuery:** ~$6.25 per TB scanned (free tier: 1 TB/month)

---

## Complete Workflow Example

### Run locally for development
```bash
python mmm_pipeline.py --data-source local
```

### Deploy to Cloud for production
```bash
# 1. Deploy function
gcloud functions deploy mmm_pipeline_function \
  --runtime python311 \
  --trigger-http \
  --allow-unauthenticated

# 2. Schedule to run daily at 2 AM
gcloud scheduler jobs create pubsub mmm-daily \
  --schedule="0 2 * * *" \
  --topic=mmm-pipeline-trigger

# 3. Monitor results
gcloud functions logs read mmm_pipeline_function --follow
```

### Access results programmatically
```python
import requests
import os

# Get Cloud Function URL
function_url = os.environ.get('MMM_FUNCTION_URL')

# Trigger pipeline
response = requests.post(function_url, json={
    'data_source': 'gcp',
    'model_type': 'ridge'
})

# Get results
results = response.json()
print(f"R²: {results['results']['model']['metrics']['r2_score']:.4f}")
```

---

## Troubleshooting

### "Permission denied" errors
```bash
# Add required IAM roles
gcloud projects add-iam-policy-binding alterdata-rekrutacja-20 \
  --member=serviceAccount:mmm-function@alterdata-rekrutacja-20.iam.gserviceaccount.com \
  --role=roles/bigquery.dataEditor

gcloud projects add-iam-policy-binding alterdata-rekrutacja-20 \
  --member=serviceAccount:mmm-function@alterdata-rekrutacja-20.iam.gserviceaccount.com \
  --role=roles/storage.objectAdmin
```

### Function times out
- Increase timeout: `--timeout 600s` (max 540s for HTTP, 3600s for Cloud Run)
- Use Cloud Run instead of Cloud Functions for longer jobs

### Out of memory
- Increase memory: `--memory 2GB` or `--memory 4GB`
- Reduce data size or process in batches

---

## Next Steps

1. **Start locally:** `python mmm_pipeline.py`
2. **Test Cloud Function:** Deploy to Cloud Functions and call via HTTP
3. **Schedule runs:** Set up Cloud Scheduler for daily/weekly execution
4. **Monitor:** Set up alerting and logging
5. **Optimize:** Adjust model types, hyperparameters, and schedule based on results
