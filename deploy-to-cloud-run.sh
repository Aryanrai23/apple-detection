#!/bin/bash
# Script to deploy Apple Harvest Management System to Google Cloud Run

# Exit on any error
set -e

# Default values
PROJECT_ID=""
REGION="us-central1"
SERVICE_NAME="apple-harvest-app"
IMAGE_NAME="apple-harvest-app"
VERSION="v1"
BUCKET_NAME="apple-454418-detection-images"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --project-id)
      PROJECT_ID="$2"
      shift # past argument
      shift # past value
      ;;
    --region)
      REGION="$2"
      shift # past argument
      shift # past value
      ;;
    --service-name)
      SERVICE_NAME="$2"
      shift # past argument
      shift # past value
      ;;
    --image-name)
      IMAGE_NAME="$2"
      shift # past argument
      shift # past value
      ;;
    --version)
      VERSION="$2"
      shift # past argument
      shift # past value
      ;;
    --bucket-name)
      BUCKET_NAME="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if project ID is provided
if [ -z "$PROJECT_ID" ]; then
  echo "Error: --project-id is required"
  echo "Usage: $0 --project-id YOUR_PROJECT_ID [--region REGION] [--service-name SERVICE_NAME] [--image-name IMAGE_NAME] [--version VERSION] [--bucket-name BUCKET_NAME]"
  exit 1
fi

echo "Deploying Apple Harvest Management System to Google Cloud Run"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service Name: $SERVICE_NAME"
echo "Image Name: $IMAGE_NAME"
echo "Version: $VERSION"
echo "Bucket Name: $BUCKET_NAME"

# Set Google Cloud project
echo "Setting Google Cloud project..."
gcloud config set project $PROJECT_ID

# Enable required services
echo "Enabling required Google Cloud services..."
gcloud services enable cloudbuild.googleapis.com run.googleapis.com storage-api.googleapis.com secretmanager.googleapis.com

# Create storage bucket if it doesn't exist
echo "Creating storage bucket if it doesn't exist..."
if ! gcloud storage buckets describe gs://$BUCKET_NAME &>/dev/null; then
  echo "Creating bucket gs://$BUCKET_NAME..."
  gcloud storage buckets create gs://$BUCKET_NAME --location=$REGION
else
  echo "Bucket gs://$BUCKET_NAME already exists."
fi

# Build and push Docker image
echo "Building Docker image..."
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME:$VERSION --build-arg BUCKET_NAME=$BUCKET_NAME .

echo "Configuring Docker to use Google Container Registry..."
gcloud auth configure-docker

echo "Pushing image to Google Container Registry..."
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME:$VERSION

# Deploy to Google Cloud Run
echo "Deploying to Google Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$IMAGE_NAME:$VERSION \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --set-env-vars="GOOGLE_APPLICATION_CREDENTIALS=/app/src/google_credentials.json,BUCKET_NAME=$BUCKET_NAME"

# Print the deployed service URL
echo "Deployment complete!"
echo "Service URL: $(gcloud run services describe $SERVICE_NAME --region $REGION --format='value(status.url)')"
echo ""
echo "IMPORTANT: Make sure to set up the GOOGLE_API_KEY as a secret in Google Cloud Secret Manager"
echo "For instructions, see the gcp-deployment-guide.md file" 