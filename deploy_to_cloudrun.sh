#!/bin/bash

# MMM Pipeline - Cloud Run Deployment Script
# This script builds and deploys the pipeline to Google Cloud Run

set -e  # Exit on error

echo "=========================================="
echo "MMM Pipeline - Cloud Run Deployment"
echo "=========================================="

# Configuration
PROJECT_ID="alterdata-rekrutacja-20"
REGION="us-central1"
IMAGE_NAME="mmm-pipeline"
SERVICE_NAME="mmm-pipeline"

# Step 1: Check prerequisites
echo ""
echo "Step 1: Checking prerequisites..."

if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Install from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "✓ gcloud and docker available"

# Step 2: Configure gcloud
echo ""
echo "Step 2: Configuring gcloud..."
gcloud config set project $PROJECT_ID
gcloud auth configure-docker $REGION-docker.pkg.dev
echo "✓ gcloud configured"

# Step 3: Enable required APIs
echo ""
echo "Step 3: Enabling required GCP APIs..."
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    bigquery.googleapis.com \
    storage.googleapis.com
echo "✓ APIs enabled"

# Step 4: Create Artifact Registry repository (if not exists)
echo ""
echo "Step 4: Setting up Artifact Registry..."
REPO_EXISTS=$(gcloud artifacts repositories describe cloud-run-source \
    --location=$REGION --quiet 2>/dev/null || echo "false")

if [ "$REPO_EXISTS" == "false" ]; then
    echo "Creating repository..."
    gcloud artifacts repositories create cloud-run-source \
        --repository-format=docker \
        --location=$REGION
fi
echo "✓ Repository ready"

# Step 5: Build Docker image
echo ""
echo "Step 5: Building Docker image..."
IMAGE_URL="$REGION-docker.pkg.dev/$PROJECT_ID/cloud-run-source/$IMAGE_NAME"
docker build -t $IMAGE_URL .
echo "✓ Image built: $IMAGE_URL"

# Step 6: Push to Artifact Registry
echo ""
echo "Step 6: Pushing image to Artifact Registry..."
docker push $IMAGE_URL
echo "✓ Image pushed"

# Step 7: Deploy to Cloud Run
echo ""
echo "Step 7: Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image=$IMAGE_URL \
    --platform=managed \
    --region=$REGION \
    --memory=2Gi \
    --cpu=2 \
    --timeout=600s \
    --no-allow-unauthenticated \
    --set-env-vars="GCP_PROJECT=$PROJECT_ID"

echo ""
echo "=========================================="
echo "✓ Deployment complete!"
echo "=========================================="
echo ""
echo "Service URL:"
gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)'
echo ""
echo "To trigger the pipeline:"
echo "  gcloud run jobs create-on-demand $SERVICE_NAME"
echo ""
echo "Or for scheduled runs, use Cloud Scheduler:"
echo "  gcloud scheduler jobs create cloud-run mmm-pipeline-daily \\"
echo "    --schedule='0 2 * * *' \\"
echo "    --http-method=POST \\"
echo "    --uri=<SERVICE_URL> \\"
echo "    --oidc-service-account-email=<YOUR_SERVICE_ACCOUNT>"
echo ""
