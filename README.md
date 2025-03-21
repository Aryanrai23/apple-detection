# Apple Harvest Management System with Google Cloud Integration

This system detects apples using computer vision, predicts their quality, uploads the data to Google Cloud Storage, and displays harvest analytics in a Streamlit dashboard.

## Features

### Detection (detection.py)
- Real-time apple detection using YOLOv5
- Cropping and saving of detected apples
- Object tracking to avoid duplicate detections
- Automatic upload to Google Cloud Storage
- JSON data export with metadata

### Quality Prediction
- Apple quality classification using TensorFlow model
- Categories: Good, Mixed, Bad quality
- Confidence scores for each quality category
- Integration with detection results

### Harvest Dashboard (app.py)
- Comprehensive harvest data visualization
- Quality distribution analytics
- Filtering and sorting of detected apples
- Quality-based recommendations for farmers
- Timeline visualization of harvest process
- Interactive charts and visualizations

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Ensure your Google Cloud credentials are in the correct location:
   ```
   src/google_credentials.json
   ```

3. Make sure you have a webcam connected to your computer.

4. Place the apple quality model in the project root:
   ```
   apple_quality_model.h5
   ```

## Running the System

### Step 1: Start the Apple Detection

Run the detection script to start capturing and processing video from your webcam:

```
python detection.py
```

This will:
- Start your webcam
- Detect apples in the video feed
- Save cropped images of detected apples locally
- Upload images to Google Cloud Storage
- Display detection results in real-time

Press 'q' to exit the detection process.

### Step 2: View the Harvest Dashboard

After running the detection, start the Streamlit dashboard:

```
streamlit run app.py
```

This will open a web browser with the dashboard where you can:
- View harvest summary metrics and quality distribution
- Analyze the quality of detected apples
- Filter apples by quality categories
- Get harvest recommendations based on quality analysis
- View raw detection and quality data

## Google Cloud Integration

The system uses Google Cloud Storage to:
- Store cropped apple images
- Save session data in JSON format
- Enable dashboard visualization from any device
- Provide persistent storage of detection results

## Quality Model

The system uses a TensorFlow model (`apple_quality_model.h5`) to predict apple quality with three categories:
- **Good**: High-quality apples suitable for direct retail
- **Mixed**: Medium-quality apples that may require sorting
- **Bad**: Low-quality apples that may need processing or disposal

If the model file is not available, a placeholder model will be created for demonstration purposes.

## Troubleshooting

- If you encounter errors with Google Cloud authentication, make sure:
  - Your credentials file is correctly located
  - The service account has the necessary permissions
  - The bucket name is correct and accessible

- If the webcam doesn't start, check:
  - Your webcam is properly connected
  - You have the necessary permissions to access the webcam
  - No other application is using the webcam

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

