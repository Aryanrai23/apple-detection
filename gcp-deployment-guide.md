# Apple Harvest Management System: Google Cloud Run Deployment Guide

This guide walks you through deploying the Apple Harvest Management System Streamlit application to Google Cloud Run.

## Prerequisites

1. [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed
2. Docker installed
3. Google Cloud account with billing enabled
4. Service account credentials with the following permissions:
   - Storage Admin (for Google Cloud Storage)
   - Cloud Run Admin
   - Service Account User

## Step 1: Set Up Google Cloud Project

1. Create a new Google Cloud project (or use an existing one):
   ```bash
   gcloud projects create [PROJECT_ID] --name="Apple Harvest Management"
   gcloud config set project [PROJECT_ID]
   ```

2. Enable the required APIs:
   ```bash
   gcloud services enable cloudbuild.googleapis.com run.googleapis.com storage-api.googleapis.com
   ```

## Step 2: Create Google Cloud Storage Bucket

Choose a unique bucket name. We recommend including your project ID in the name:

```bash
gcloud storage buckets create gs://[PROJECT_ID]-detection-images --location=us-central1
```

For example:
```bash
gcloud storage buckets create gs://apple-454418-detection-images --location=us-central1
```

## Step 3: Set Up Google Cloud Service Account

1. Create a service account:
   ```bash
   gcloud iam service-accounts create apple-app-sa --display-name="Apple App Service Account"
   ```

2. Grant necessary permissions:
   ```bash
   # Storage permissions
   gcloud projects add-iam-policy-binding [PROJECT_ID] \
     --member="serviceAccount:apple-app-sa@[PROJECT_ID].iam.gserviceaccount.com" \
     --role="roles/storage.admin"
   
   # Cloud Run permissions
   gcloud projects add-iam-policy-binding [PROJECT_ID] \
     --member="serviceAccount:apple-app-sa@[PROJECT_ID].iam.gserviceaccount.com" \
     --role="roles/run.admin"
   ```

3. Generate and download the service account key:
   ```bash
   gcloud iam service-accounts keys create src/google_credentials.json \
     --iam-account=apple-app-sa@[PROJECT_ID].iam.gserviceaccount.com
   ```

## Step 4: Set Up Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey) to generate an API key for Gemini
2. Create a `.env` file in the project root with your API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Step 5: Build and Push Docker Image

1. Build the Docker image with your specific bucket name:
   ```bash
   docker build -t gcr.io/[PROJECT_ID]/apple-harvest-app:v1 --build-arg BUCKET_NAME=[YOUR_BUCKET_NAME] .
   ```

   Example:
   ```bash
   docker build -t gcr.io/apple-454418/apple-harvest-app:v1 --build-arg BUCKET_NAME=apple-454418-detection-images .
   ```

2. Configure Docker to use Google Container Registry:
   ```bash
   gcloud auth configure-docker
   ```

3. Push the image to Google Container Registry:
   ```bash
   docker push gcr.io/[PROJECT_ID]/apple-harvest-app:v1
   ```

## Step 6: Deploy to Google Cloud Run

```bash
gcloud run deploy apple-harvest-app \
  --image gcr.io/[PROJECT_ID]/apple-harvest-app:v1 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --service-account apple-app-sa@[PROJECT_ID].iam.gserviceaccount.com \
  --set-env-vars="GOOGLE_APPLICATION_CREDENTIALS=/app/src/google_credentials.json,BUCKET_NAME=[YOUR_BUCKET_NAME]"
```

Example:
```bash
gcloud run deploy apple-harvest-app \
  --image gcr.io/apple-454418/apple-harvest-app:v1 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --service-account apple-app-sa@apple-454418.iam.gserviceaccount.com \
  --set-env-vars="GOOGLE_APPLICATION_CREDENTIALS=/app/src/google_credentials.json,BUCKET_NAME=apple-454418-detection-images"
```

## Step 7: Set Up Secure Environment Variables

For the Gemini API key, use Google Cloud Run's secret management:

1. Create a secret in Secret Manager:
   ```bash
   echo -n "your_api_key_here" | gcloud secrets create gemini-api-key --data-file=-
   ```

2. Grant access to the service account:
   ```bash
   gcloud secrets add-iam-policy-binding gemini-api-key \
     --member="serviceAccount:apple-app-sa@[PROJECT_ID].iam.gserviceaccount.com" \
     --role="roles/secretmanager.secretAccessor"
   ```

3. Update the Cloud Run service to use the secret:
   ```bash
   gcloud run services update apple-harvest-app \
     --update-secrets=GOOGLE_API_KEY=gemini-api-key:latest
   ```

## Step 8: Verify Deployment

After deployment, Google Cloud Run will provide a URL for your application. Open this URL in your browser to access the Streamlit dashboard.

## Troubleshooting

### Authentication Issues
If you encounter authentication issues:
1. Verify the service account has the correct permissions
2. Check that the credentials file is correctly mounted in the container
3. Make sure the environment variables are correctly set

### Streamlit Issues
If there are issues with the Streamlit application:
1. Check Cloud Run logs for errors
2. Verify that all required dependencies are installed
3. Make sure the app is correctly configured to run on port 8501

## Cost Management

To manage costs:
1. Use the smallest Cloud Run instance that meets your needs
2. Consider setting minimum instances to 0 to scale to zero when not in use
3. Monitor your Google Cloud Storage usage and clean up old files if needed

## Continuous Deployment

For continuous deployment, consider setting up a Cloud Build trigger that automatically builds and deploys your application when you push to your repository. 