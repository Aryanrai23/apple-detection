#!/bin/bash
set -e

# Make sure models directory exists
mkdir -p src/models

# Check if model files exist
echo "Checking model files..."
ls -la src/models/

# If models exist, show their sizes
if [ -f "src/models/leaf_model.h5" ]; then
    echo "Leaf model exists, size: $(du -h src/models/leaf_model.h5 | cut -f1)"
else
    echo "WARNING: leaf_model.h5 not found in src/models/ directory!"
fi

if [ -f "src/models/apple_quality_model.h5" ]; then
    echo "Apple quality model exists, size: $(du -h src/models/apple_quality_model.h5 | cut -f1)"
else
    echo "WARNING: apple_quality_model.h5 not found in src/models/ directory!"
fi

# Deploy using gcloud run deploy with the source flag
echo "Starting deployment..."
gcloud run deploy apple-harvest-app \
  --source . \
  --region us-central1 \
  --allow-unauthenticated

echo "Deployment complete!" 