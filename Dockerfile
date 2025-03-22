FROM python:3.9-slim

WORKDIR /app

# Accept build argument for bucket name
ARG BUCKET_NAME=apple-454418-detection-images
ENV BUCKET_NAME=$BUCKET_NAME

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/src/models

# Copy application code
COPY . .

# Ensure the models directory exists
RUN mkdir -p /app/src/models && \
    ls -la /app/src/ && \
    ls -la /app/src/models || echo "Models directory is empty"

# Create directory and copy the Google credentials file to a standard location
RUN mkdir -p /etc/google/auth/ && \
    cp /app/src/google_credentials.json /etc/google/auth/credentials.json && \
    chmod 600 /etc/google/auth/credentials.json

# Set environment variable for Google Cloud to use the absolute path
ENV GOOGLE_APPLICATION_CREDENTIALS="/etc/google/auth/credentials.json"

# Set the default port that Streamlit will run on
ENV PORT 8080

# Use the CMD directive to specify the command to run
CMD streamlit run src/app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false