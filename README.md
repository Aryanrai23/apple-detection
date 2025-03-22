# Apple Harvest Management System

A comprehensive dashboard for monitoring apple harvest quality and disease detection.

## Features

- Real-time apple quality analysis using machine learning
- Disease detection on apple fruit images
- Leaf disease analysis capabilities
- Harvest statistics and reporting
- Integration with Google Cloud Storage for image management

## Setup Instructions

### Prerequisites

- Python 3.9+
- Google Cloud account with Storage access
- Service account with appropriate permissions

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/YourUsername/apple-detection.git
   cd apple-detection
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up Google Cloud credentials:
   - Create a service account in Google Cloud Console
   - Grant it the necessary permissions (Storage Object Viewer at minimum)
   - Download the JSON credentials file
   - Rename it to `google_credentials.json` and place it in the `src/` directory
   - ⚠️ Do not commit this file to version control!

4. Configure the application:
   - Set the `BUCKET_NAME` environment variable to your GCS bucket name
   - Or update it in the Dockerfile

### Local Development

Run the application locally:
```
streamlit run src/app.py
```

### Deployment to Google Cloud Run

1. Make sure you have the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and configured.

2. Deploy using the provided script:
   ```
   chmod +x deploy.sh
   ./deploy.sh
   ```

3. Or deploy manually:
   ```
   gcloud run deploy apple-harvest-app \
     --source . \
     --region us-central1 \
     --allow-unauthenticated
   ```

## Machine Learning Models

This application uses two main TensorFlow models:

1. `apple_quality_model.h5` - For detecting apple diseases and quality
2. `leaf_model.h5` - For analyzing leaf diseases

The models should be placed in the `src/models/` directory before deployment.

## Directory Structure

```
.
├── Dockerfile                 # Container configuration
├── README.md                  # This file
├── cloudbuild.yaml            # Cloud Build configuration
├── deploy.sh                  # Deployment script
├── requirements.txt           # Python dependencies
└── src/
    ├── app.py                 # Main Streamlit application
    ├── google_credentials_sample.json  # Sample credentials (for reference)
    ├── leaf_disease_prediction.py  # Leaf disease prediction module
    ├── models/                # ML models directory
    │   ├── apple_quality_model.h5
    │   └── leaf_model.h5
    └── quality_prediction.py  # Apple quality prediction module
```

## Security Considerations

- Never commit your Google Cloud credentials to version control
- The provided credential sample is for reference only
- Use environment variables when possible
- Follow the principle of least privilege for service accounts

## License

[MIT License](LICENSE)

---

