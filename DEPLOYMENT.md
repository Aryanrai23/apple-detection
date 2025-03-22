# Streamlit Deployment to Google Cloud Run

This document provides a simplified guide for deploying the Apple Harvest Management Streamlit application to Google Cloud Run.

## Quick Start

1. Make sure you have the following prerequisites:
   - Google Cloud SDK installed
   - Docker installed
   - Google Cloud account with billing enabled

2. Clone this repository:
   ```bash
   git clone [your-repository-url]
   cd apple-detection
   ```

3. Create or update your Google Cloud service account credentials:
   - Create a service account with Storage Admin and Cloud Run Admin permissions
   - Download the credentials JSON file
   - Place it in `src/google_credentials.json`

4. Create a unique storage bucket (include your project ID for uniqueness):
   ```bash
   gcloud storage buckets create gs://[PROJECT_ID]-detection-images --location=us-central1
   ```
   
   Example:
   ```bash
   gcloud storage buckets create gs://apple-454418-detection-images --location=us-central1
   ```

5. Set up your Gemini API key:
   - Create a `.env` file with your GOOGLE_API_KEY

6. Run the deployment script with your bucket name:
   ```bash
   chmod +x deploy-to-cloud-run.sh
   ./deploy-to-cloud-run.sh --project-id YOUR_PROJECT_ID --bucket-name YOUR_BUCKET_NAME
   ```
   
   Example:
   ```bash
   ./deploy-to-cloud-run.sh --project-id apple-454418 --bucket-name apple-454418-detection-images
   ```

## Streamlit Application

The Apple Harvest Management System is a Streamlit-based web application that:
- Displays apple harvest data visualizations
- Integrates with Google Cloud Storage for image data
- Uses the Gemini API for intelligent analysis
- Features a quality prediction model to assess apple quality

## Manual Deployment Steps

If you prefer to deploy manually instead of using the script:

1. Build and push the Docker image:
   ```bash
   docker build -t gcr.io/[PROJECT_ID]/apple-harvest-app:v1 --build-arg BUCKET_NAME=[YOUR_BUCKET_NAME] .
   gcloud auth configure-docker
   docker push gcr.io/[PROJECT_ID]/apple-harvest-app:v1
   ```

2. Deploy to Google Cloud Run:
   ```bash
   gcloud run deploy apple-harvest-app \
     --image gcr.io/[PROJECT_ID]/apple-harvest-app:v1 \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 1Gi \
     --cpu 1 \
     --set-env-vars="GOOGLE_APPLICATION_CREDENTIALS=/app/src/google_credentials.json,BUCKET_NAME=[YOUR_BUCKET_NAME]"
   ```

## Additional Resources

- For detailed deployment instructions, see [gcp-deployment-guide.md](gcp-deployment-guide.md)
- To deploy with a script, use [deploy-to-cloud-run.sh](deploy-to-cloud-run.sh) 