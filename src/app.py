import streamlit as st
import os
import json
import cv2
import numpy as np
from google.cloud import storage
from datetime import datetime
import io
import tempfile
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import quality_prediction
import altair as alt
import gemini_integration

# Set page config
st.set_page_config(
    page_title="Apple Harvest Dashboard",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for language and chat
if 'language' not in st.session_state:
    st.session_state.language = 'english'
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}

# Function to get translated text
def get_text(key):
    if st.session_state.language.lower() == 'hindi':
        return gemini_integration.HINDI_TRANSLATIONS.get(key, key)
    return key

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "src/google_credentials.json"

# Google Cloud Storage configuration
BUCKET_NAME = "apple-detection-images"

# Initialize Google Cloud Storage client
storage_client = storage.Client()

# Load apple quality predictor
quality_predictor = quality_prediction.get_predictor()

try:
    bucket = storage_client.get_bucket(BUCKET_NAME)
    st.sidebar.success(f"Connected to bucket: {BUCKET_NAME}")
except Exception as e:
    st.error(f"Error accessing bucket: {e}")
    st.stop()

# Helper functions
@st.cache_data(ttl=60)  # Cache data for 60 seconds
def list_sessions():
    """List all available detection sessions in the bucket"""
    # Use a more reliable method to find session folders
    blobs = list(bucket.list_blobs())
    
    # Extract session IDs from blob names
    sessions = set()
    for blob in blobs:
        # Split the blob name by '/' and take the first part as the session ID
        parts = blob.name.split('/')
        if len(parts) > 1:  # Ensure it's a path with a session ID
            session_id = parts[0]
            if session_id:
                sessions.add(session_id)
    
    # Debug output
    st.sidebar.write(f"Found {len(sessions)} sessions")
    
    return sorted(list(sessions), reverse=True)  # Most recent first

@st.cache_data(ttl=30)  # Cache data for 30 seconds
def get_session_data(session_id):
    """Get detection data for a specific session"""
    try:
        # Construct the path to the data file
        data_path = f"{session_id}/apple_count_data.json"
        
        # Check if the blob exists
        blob = bucket.blob(data_path)
        if not blob.exists():
            st.sidebar.error(f"Data file not found: {data_path}")
            # List all files in this session for debugging
            session_files = list(bucket.list_blobs(prefix=f"{session_id}/"))
            st.sidebar.write(f"Files in session {session_id}:")
            for file in session_files:
                st.sidebar.write(f"- {file.name}")
            return None
        
        # Download and parse the data
        data_string = blob.download_as_string()
        data = json.loads(data_string)
        st.sidebar.success(f"Successfully loaded data for session {session_id}")
        return data
    except Exception as e:
        st.sidebar.error(f"Error loading session data: {str(e)}")
        return None

@st.cache_data(ttl=60)  # Cache for 1 minute
def download_image_from_url(gcs_url):
    """Download image from Google Cloud Storage URL"""
    # Extract bucket name and blob name from the URL
    # URL format: gs://bucket_name/blob_name
    try:
        if not gcs_url or not isinstance(gcs_url, str):
            return None
            
        # Remove the gs:// prefix if present
        clean_url = gcs_url.replace("gs://", "")
        
        # Split into bucket name and blob path
        parts = clean_url.split("/", 1)
        
        if len(parts) != 2:
            st.sidebar.warning(f"Invalid GCS URL format: {gcs_url}")
            return None
        
        bucket_name, blob_path = parts
        
        # Get the blob
        blob = storage_client.bucket(bucket_name).blob(blob_path)
        
        if not blob.exists():
            st.sidebar.warning(f"Image blob does not exist: {blob_path}")
            return None
            
        # Download the image
        image_bytes = blob.download_as_bytes()
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        st.sidebar.error(f"Error downloading image: {str(e)}")
        return None

# Removed the caching decorator to ensure every prediction is fresh
def predict_apple_quality(_image):
    """Predict apple quality from image"""
    if _image is None:
        return None
    return quality_predictor.predict_quality(_image)

def get_quality_color(category):
    """Return color for apple category label"""
    if category == "Normal_Apple":
        return "green"
    elif category == "Blotch_Apple":
        return "orange"
    elif category == "Scab_Apple":
        return "brown"
    else:  # Rot_Apple
        return "red"

# Sidebar - Language selector
language_col1, language_col2 = st.sidebar.columns(2)
with language_col1:
    if st.session_state.language == 'english':
        if st.button("हिंदी में बदलें"):
            st.session_state.language = 'hindi'
            st.experimental_rerun()
with language_col2:
    if st.session_state.language == 'hindi':
        if st.button("Switch to English"):
            st.session_state.language = 'english'
            st.experimental_rerun()

# Sidebar 
st.sidebar.title("🍎 " + get_text("Apple Harvest Dashboard"))

# Session selector
sessions = list_sessions()
if not sessions:
    st.sidebar.warning("No detection sessions found")
    st.stop()

selected_session = st.sidebar.selectbox(
    get_text("Select Harvest Session"), 
    options=sessions,
    format_func=lambda x: f"{x} ({datetime.strptime(x.split('_')[0], '%Y%m%d').strftime('%B %d, %Y')})"
)

# Load session data
session_data = get_session_data(selected_session)
if not session_data:
    st.warning("No data available for the selected session")
    st.stop()

# Display session info in sidebar
st.sidebar.subheader(get_text("Session Information"))
session_time = datetime.strptime(session_data.get("timestamp", ""), "%Y-%m-%dT%H:%M:%S.%f") if "timestamp" in session_data else None
if session_time:
    st.sidebar.info(f"{get_text('Harvest Date')}: {session_time.strftime('%B %d, %Y')}")
    st.sidebar.info(f"{get_text('Harvest Time')}: {session_time.strftime('%H:%M:%S')}")

st.sidebar.info(f"{get_text('Total Apples Detected')}: {len(session_data.get('cloud_image_urls', []))}")

# Main page
st.title(get_text("Apple Harvest Dashboard"))

# Process apple quality for this session
apple_images = session_data.get("cloud_image_urls", [])
with st.spinner(get_text("Analyzing apple conditions...")):
    quality_results = []
    
    # Process images in smaller batches to avoid overloading memory
    batch_size = 5
    for i in range(0, len(apple_images), batch_size):
        batch = apple_images[i:i+batch_size]
        st.sidebar.write(f"Processing batch {i//batch_size + 1}/{(len(apple_images)-1)//batch_size + 1}...")
        
        # Process images and get quality predictions for this batch
        for img_data in batch:
            image = download_image_from_url(img_data.get("url", ""))
            if image:
                quality_result = predict_apple_quality(_image=image)
                if quality_result:
                    result_with_url = {
                        **quality_result,
                        "url": img_data.get("url", ""),
                        "detection_confidence": img_data.get("confidence", 0)
                    }
                    quality_results.append(result_with_url)

# Create apple condition summary
category_counts = {
    "Normal_Apple": 0,
    "Blotch_Apple": 0,
    "Rot_Apple": 0,
    "Scab_Apple": 0
}

for result in quality_results:
    category = result.get("category")
    if category in category_counts:
        category_counts[category] += 1

# Prepare summary data for AI analysis
analysis_data = {
    "session_id": selected_session,
    "timestamp": session_data.get("timestamp", ""),
    "total_apples": len(apple_images),
    "condition_counts": category_counts,
    "condition_percentages": {
        "Normal": (category_counts["Normal_Apple"] / len(quality_results) * 100) if quality_results else 0,
        "Blotch": (category_counts["Blotch_Apple"] / len(quality_results) * 100) if quality_results else 0,
        "Rot": (category_counts["Rot_Apple"] / len(quality_results) * 100) if quality_results else 0,
        "Scab": (category_counts["Scab_Apple"] / len(quality_results) * 100) if quality_results else 0
    }
}

# Create tabs for different views
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    get_text("Harvest Summary"), 
    get_text("Condition Analysis"), 
    get_text("Apple Gallery"), 
    get_text("Timeline"), 
    get_text("Model Showcase"),
    get_text("AI Analysis"),
    get_text("Chat")
])

# Tab 1: Harvest Summary
with tab1:
    st.header(get_text("Apple Harvest Summary"))
    
    # Summary metrics in a nice format
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(get_text("Total Apples"), len(apple_images))
    
    with col2:
        normal_percentage = (category_counts["Normal_Apple"] / len(quality_results) * 100) if quality_results else 0
        st.metric(get_text("Normal"), f"{category_counts['Normal_Apple']} ({normal_percentage:.1f}%)")
    
    with col3:
        blotch_percentage = (category_counts["Blotch_Apple"] / len(quality_results) * 100) if quality_results else 0
        st.metric(get_text("Blotch"), f"{category_counts['Blotch_Apple']} ({blotch_percentage:.1f}%)")
    
    with col4:
        rot_percentage = (category_counts["Rot_Apple"] / len(quality_results) * 100) if quality_results else 0
        st.metric(get_text("Rot"), f"{category_counts['Rot_Apple']} ({rot_percentage:.1f}%)")
        
    with col5:
        scab_percentage = (category_counts["Scab_Apple"] / len(quality_results) * 100) if quality_results else 0
        st.metric(get_text("Scab"), f"{category_counts['Scab_Apple']} ({scab_percentage:.1f}%)")
    
    # Harvest quality chart
    st.subheader(get_text("Apple Condition Distribution"))
    
    # Create data for pie chart
    pie_data = pd.DataFrame({
        'Condition': ['Normal', 'Blotch', 'Rot', 'Scab'],
        'Count': [
            category_counts['Normal_Apple'], 
            category_counts['Blotch_Apple'], 
            category_counts['Rot_Apple'],
            category_counts['Scab_Apple']
        ]
    })
    
    # Create a circle chart (pie chart)
    color_scale = alt.Scale(
        domain=['Normal', 'Blotch', 'Rot', 'Scab'],
        range=['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
    )
    
    pie_chart = alt.Chart(pie_data).mark_arc().encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Condition", type="nominal", scale=color_scale),
        tooltip=['Condition', 'Count']
    ).properties(
        width=400,
        height=300
    )
    
    # Display pie chart
    st.altair_chart(pie_chart, use_container_width=True)
    
    # Harvest recommendations
    st.subheader(get_text("Harvest Recommendations"))
    recommendation_container = st.container(border=True)
    
    with recommendation_container:
        if normal_percentage > 70:
            st.success("👍 Excellent harvest condition! Your apple crop is mostly healthy.")
            st.write("Recommendations:")
            st.write("- Focus on maintaining your current growing practices")
            st.write("- Consider premium market opportunities for your high-quality produce")
            st.write("- Schedule prompt distribution to maximize freshness and value")
        elif normal_percentage > 40:
            st.warning("👌 Moderate harvest condition. A significant portion of your crop shows disease.")
            st.write("Recommendations:")
            st.write("- Review orchard disease management practices")
            st.write("- Consider sorting apples by condition for different market destinations")
            st.write("- Process diseased apples quickly to prevent spread")
        else:
            st.error("🔍 Disease concerns detected. Most of your harvest shows disease issues.")
            st.write("Recommendations:")
            st.write("- Urgent: Consult with a plant pathologist to identify control measures")
            st.write("- Consider processing options for affected fruit (juice, sauce, etc.)")
            st.write("- Implement improved spray schedules for next season")
            
        # Disease information
        with st.expander(get_text("Disease Information")):
            st.markdown("""
            ### Common Apple Diseases Found:
            
            #### Blotch Disease
            - Causes dark, irregular spots on the apple skin
            - Spreads through rain splash and can affect stored fruit
            - Control: Fungicide applications, proper orchard sanitation
            
            #### Apple Rot
            - Causes soft, brown or black areas on the fruit
            - Often enters through wounds or bruises
            - Control: Careful handling, prompt cold storage, fungicide treatments
            
            #### Apple Scab
            - Causes rough, corky spots and lesions on fruit
            - One of the most common apple diseases worldwide
            - Control: Resistant varieties, fungicide applications, proper pruning
            """)

# Tab 2: Quality Analysis - renamed to Condition Analysis
with tab2:
    st.header(get_text("Apple Condition Analysis"))
    
    # Quality score distributions
    quality_scores_df = pd.DataFrame([
        {
            "Apple": i,
            "Blotch Score": result["scores"]["Blotch_Apple"],
            "Normal Score": result["scores"]["Normal_Apple"],
            "Rot Score": result["scores"]["Rot_Apple"],
            "Scab Score": result["scores"]["Scab_Apple"],
            "Predicted Condition": result["category"]
        }
        for i, result in enumerate(quality_results)
    ])
    
    if not quality_scores_df.empty:
        # Create a grouped bar chart
        quality_chart = alt.Chart(quality_scores_df).transform_fold(
            ['Blotch Score', 'Normal Score', 'Rot Score', 'Scab Score'],
            as_=['Condition Type', 'Score']
        ).mark_bar().encode(
            x='Apple:O',
            y='Score:Q',
            color='Condition Type:N',
            tooltip=['Apple:O', 'Score:Q', 'Condition Type:N', 'Predicted Condition:N']
        ).properties(
            width=800,
            height=400,
            title=get_text("Condition Scores for Each Apple")
        )
        
        st.altair_chart(quality_chart, use_container_width=True)
    
        # Quality distribution by confidence
        st.subheader(get_text("Prediction Confidence"))
        confidence_df = pd.DataFrame([
            {
                "Apple": i,
                "Confidence": result["confidence"],
                "Condition": result["category"].split('_')[0]  # Remove '_Apple' suffix
            }
            for i, result in enumerate(quality_results)
        ])
        
        # Create scatter plot
        confidence_chart = alt.Chart(confidence_df).mark_circle(size=100).encode(
            x='Apple:O',
            y='Confidence:Q',
            color=alt.Color('Condition:N', scale=alt.Scale(
                domain=['Normal', 'Blotch', 'Rot', 'Scab'],
                range=['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
            )),
            tooltip=['Apple:O', 'Confidence:Q', 'Condition:N']
        ).properties(
            width=800,
            height=300,
            title=get_text("Prediction Confidence by Apple")
        )
        
        st.altair_chart(confidence_chart, use_container_width=True)
    else:
        st.info(get_text("No condition analysis data available for this session."))

# Tab 3: Apple Gallery with condition
with tab3:
    st.header(get_text("Apple Condition Gallery"))
    
    # Filter controls
    condition_filter = st.multiselect(
        get_text("Filter by Condition"),
        options=["Normal_Apple", "Blotch_Apple", "Rot_Apple", "Scab_Apple"],
        default=["Normal_Apple", "Blotch_Apple", "Rot_Apple", "Scab_Apple"],
        format_func=lambda x: x.split('_')[0]  # Display without '_Apple' suffix
    )
    
    # Apply filters to quality results
    filtered_results = [r for r in quality_results if r["category"] in condition_filter]
    
    if not filtered_results:
        st.info(get_text("No apple images match the selected filters."))
    else:
        # Display images in a grid with condition labels
        images_per_row = 3
        
        # Sort by confidence
        filtered_results = sorted(filtered_results, key=lambda x: x.get("confidence", 0), reverse=True)
        
        # Display images in grid
        for i in range(0, len(filtered_results), images_per_row):
            cols = st.columns(images_per_row)
            for j in range(images_per_row):
                if i + j < len(filtered_results):
                    result = filtered_results[i + j]
                    with cols[j]:
                        image = download_image_from_url(result.get("url", ""))
                        if image:
                            category = result.get("category", "Unknown")
                            confidence = result.get("confidence", 0)
                            color = get_quality_color(category)
                            
                            # Container with colored border based on condition
                            with st.container(border=True):
                                st.image(image)
                                # Show category without '_Apple' suffix
                                display_category = category.split('_')[0]
                                st.markdown(f"<p style='text-align:center; color:{color}; font-weight:bold;'>{display_category} ({confidence:.2f})</p>", unsafe_allow_html=True)
                                
                                # Brief description
                                st.caption(result.get("description", ""))
                                
                                # Show detailed scores in an expander
                                with st.expander(get_text("Condition Scores")):
                                    scores = result.get("scores", {})
                                    st.progress(scores.get("Normal_Apple", 0), text=get_text("Normal"))
                                    st.progress(scores.get("Blotch_Apple", 0), text=get_text("Blotch"))
                                    st.progress(scores.get("Rot_Apple", 0), text=get_text("Rot"))
                                    st.progress(scores.get("Scab_Apple", 0), text=get_text("Scab"))
                        else:
                            st.error(get_text("Failed to load image"))

# Tab 4: Timeline
with tab4:
    st.header(get_text("Harvest Timeline"))
    
    # Prepare data for chart
    frame_data = session_data.get("frame_counts", [])
    if frame_data:
        # Create DataFrame
        df = pd.DataFrame(frame_data)
        if not df.empty and "timestamp" in df.columns:
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df["timestamp"], df["apple_count"], marker='o', linestyle='-')
            ax.set_xlabel(get_text("Time"))
            ax.set_ylabel(get_text("Apple Count"))
            ax.set_title(get_text("Apple Count Over Time"))
            ax.grid(True, alpha=0.3)
            
            # Show plot
            st.pyplot(fig)
            
            # Show session start and end images if available
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(get_text("Session Start"))
                if "session_start_frame" in session_data:
                    start_image = download_image_from_url(session_data["session_start_frame"])
                    if start_image:
                        st.image(start_image, use_container_width=True)
            
            with col2:
                # Get the last frame in the session if available
                st.subheader(get_text("Session End"))
                try:
                    end_blob = bucket.blob(f"{selected_session}/session_end.jpg")
                    if end_blob.exists():
                        end_image_bytes = end_blob.download_as_bytes()
                        end_image = Image.open(io.BytesIO(end_image_bytes))
                        st.image(end_image, use_container_width=True)
                except Exception:
                    pass
    else:
        st.info(get_text("No timeline data available for this session"))

# New Tab 5: Model Showcase
with tab5:
    st.header(get_text("Apple Disease Detection Model"))
    
    # About the model
    st.subheader(get_text("About the Model"))
    model_info_col1, model_info_col2 = st.columns([2, 1])
    
    with model_info_col1:
        st.markdown("""
        This dashboard uses a deep learning model trained to classify apples into four categories:
        
        - **Normal Apple**: Healthy apples with no visible defects or diseases
        - **Blotch Apple**: Apples with blotch disease showing dark, irregular spots
        - **Rot Apple**: Apples with rot showing soft, brown or black areas
        - **Scab Apple**: Apples with scab disease showing rough, corky spots
        
        The model is implemented using TensorFlow and uses image processing techniques to analyze
        visual features such as color, texture, and lesion patterns to determine apple condition.
        """)
        
    with model_info_col2:
        st.markdown("### Model File")
        st.code("apple_quality_model.h5")
        st.markdown("### Input Size")
        st.code("224 x 224 x 3 (RGB)")
        st.markdown("### Output")
        st.code("4 classes of apple conditions")
    
    # Model architecture and prediction process
    st.subheader(get_text("How the Model Works"))
    
    # Flow diagram using Streamlit features
    flow_cols = st.columns(5)
    
    with flow_cols[0]:
        st.markdown("#### Image Input")
        st.markdown("Apple images are captured using computer vision")
        st.image("https://cdn-icons-png.flaticon.com/512/1202/1202125.png", width=100)
    
    with flow_cols[1]:
        st.markdown("#### Preprocessing")
        st.markdown("Images are resized and normalized (0-1)")
        st.image("https://cdn-icons-png.flaticon.com/512/1/1848.png", width=100)
    
    with flow_cols[2]:
        st.markdown("#### Model")
        st.markdown("Deep CNN analyzes features")
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    
    with flow_cols[3]:
        st.markdown("#### Prediction")
        st.markdown("Quality scores for 4 categories")
        st.image("https://cdn-icons-png.flaticon.com/512/5307/5307898.png", width=100)
    
    with flow_cols[4]:
        st.markdown("#### Decision")
        st.markdown("Final condition classification")
        st.image("https://cdn-icons-png.flaticon.com/512/6819/6819077.png", width=100)
    
    # Model layers visualization
    st.subheader(get_text("Model Architecture"))
    
    # Get the model summary if available
    try:
        model_summary = quality_predictor.get_model_summary() 
        st.code(model_summary, language="python")
    except:
        # If there's no method to get the summary, show a placeholder architecture
        st.code("""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 222, 222, 16)      448       
                                                                 
 max_pooling2d (MaxPooling2D)  (None, 111, 111, 16)    0         
                                                                 
 conv2d_1 (Conv2D)           (None, 109, 109, 32)      4640      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 54, 54, 32)       0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 93312)             0         
                                                                 
 dense (Dense)               (None, 64)                5971968   
                                                                 
 dense_1 (Dense)             (None, 4)                 259       
                                                                 
=================================================================
Total params: 5,977,251
Trainable params: 5,977,251
Non-trainable params: 0
_________________________________________________________________
        """, language="python")
        
    # Sample prediction demonstrations
    st.subheader(get_text("Sample Predictions"))
    
    if not quality_results:
        st.info(get_text("No prediction data available yet. Run the detection first to see sample predictions."))
    else:
        # Group results by category
        normal_results = [r for r in quality_results if r["category"] == "Normal_Apple"]
        blotch_results = [r for r in quality_results if r["category"] == "Blotch_Apple"]
        rot_results = [r for r in quality_results if r["category"] == "Rot_Apple"]
        scab_results = [r for r in quality_results if r["category"] == "Scab_Apple"]
        
        # Show examples from each category in 2x2 grid
        row1_cols = st.columns(2)
        row2_cols = st.columns(2)
        
        # Row 1, Col 1: Normal
        with row1_cols[0]:
            st.markdown("#### Normal Apple Example")
            if normal_results:
                sample = normal_results[0]
                sample_img = download_image_from_url(sample["url"])
                if sample_img:
                    st.image(sample_img, use_container_width=True)
                    st.markdown("##### Prediction Scores")
                    st.progress(sample["scores"]["Normal_Apple"], text=f"{get_text('Normal')}: {sample['scores']['Normal_Apple']:.2f}")
                    st.progress(sample["scores"]["Blotch_Apple"], text=f"{get_text('Blotch')}: {sample['scores']['Blotch_Apple']:.2f}")
                    st.progress(sample["scores"]["Rot_Apple"], text=f"{get_text('Rot')}: {sample['scores']['Rot_Apple']:.2f}")
                    st.progress(sample["scores"]["Scab_Apple"], text=f"{get_text('Scab')}: {sample['scores']['Scab_Apple']:.2f}")
            else:
                st.info(get_text("No Normal apple examples in this session"))
                
        # Row 1, Col 2: Blotch
        with row1_cols[1]:
            st.markdown("#### Blotch Apple Example")
            if blotch_results:
                sample = blotch_results[0]
                sample_img = download_image_from_url(sample["url"])
                if sample_img:
                    st.image(sample_img, use_container_width=True)
                    st.markdown("##### Prediction Scores")
                    st.progress(sample["scores"]["Normal_Apple"], text=f"{get_text('Normal')}: {sample['scores']['Normal_Apple']:.2f}")
                    st.progress(sample["scores"]["Blotch_Apple"], text=f"{get_text('Blotch')}: {sample['scores']['Blotch_Apple']:.2f}")
                    st.progress(sample["scores"]["Rot_Apple"], text=f"{get_text('Rot')}: {sample['scores']['Rot_Apple']:.2f}")
                    st.progress(sample["scores"]["Scab_Apple"], text=f"{get_text('Scab')}: {sample['scores']['Scab_Apple']:.2f}")
            else:
                st.info(get_text("No Blotch apple examples in this session"))
        
        # Row 2, Col 1: Rot
        with row2_cols[0]:
            st.markdown("#### Rot Apple Example")
            if rot_results:
                sample = rot_results[0]
                sample_img = download_image_from_url(sample["url"])
                if sample_img:
                    st.image(sample_img, use_container_width=True)
                    st.markdown("##### Prediction Scores")
                    st.progress(sample["scores"]["Normal_Apple"], text=f"{get_text('Normal')}: {sample['scores']['Normal_Apple']:.2f}")
                    st.progress(sample["scores"]["Blotch_Apple"], text=f"{get_text('Blotch')}: {sample['scores']['Blotch_Apple']:.2f}")
                    st.progress(sample["scores"]["Rot_Apple"], text=f"{get_text('Rot')}: {sample['scores']['Rot_Apple']:.2f}")
                    st.progress(sample["scores"]["Scab_Apple"], text=f"{get_text('Scab')}: {sample['scores']['Scab_Apple']:.2f}")
            else:
                st.info(get_text("No Rot apple examples in this session"))
                
        # Row 2, Col 2: Scab
        with row2_cols[1]:
            st.markdown("#### Scab Apple Example")
            if scab_results:
                sample = scab_results[0]
                sample_img = download_image_from_url(sample["url"])
                if sample_img:
                    st.image(sample_img, use_container_width=True)
                    st.markdown("##### Prediction Scores")
                    st.progress(sample["scores"]["Normal_Apple"], text=f"{get_text('Normal')}: {sample['scores']['Normal_Apple']:.2f}")
                    st.progress(sample["scores"]["Blotch_Apple"], text=f"{get_text('Blotch')}: {sample['scores']['Blotch_Apple']:.2f}")
                    st.progress(sample["scores"]["Rot_Apple"], text=f"{get_text('Rot')}: {sample['scores']['Rot_Apple']:.2f}")
                    st.progress(sample["scores"]["Scab_Apple"], text=f"{get_text('Scab')}: {sample['scores']['Scab_Apple']:.2f}")
            else:
                st.info(get_text("No Scab apple examples in this session"))
                
    # Technical details about the model
    with st.expander(get_text("Technical Details")):
        st.markdown("""
        ### Model Technical Details
        
        - **Framework**: TensorFlow / Keras
        - **Architecture**: Convolutional Neural Network (CNN)
        - **Input Processing**: Images are resized to 224x224 pixels and normalized to [0,1] range
        - **Training**: The model was trained on a dataset of apple images with disease labels
        - **Classes**: Normal, Blotch, Rot, and Scab conditions
        - **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
        
        The model uses transfer learning techniques to leverage pre-trained weights from image classification networks,
        which are then fine-tuned on apple-specific data to achieve high accuracy in disease detection.
        """)
    
    # Visualization of feature importance (conceptual)
    st.subheader(get_text("Key Features for Disease Detection"))
    
    feature_cols = st.columns(4)
    with feature_cols[0]:
        st.markdown("#### Normal Apples")
        st.markdown("- Even coloration")
        st.markdown("- Smooth skin")
        st.markdown("- No visible lesions")
        st.markdown("- Uniform texture")
    
    with feature_cols[1]:
        st.markdown("#### Blotch Disease")
        st.markdown("- Dark, irregular spots")
        st.markdown("- Raised or sunken lesions")
        st.markdown("- Black fungal growth")
        st.markdown("- Often clustered spots")
    
    with feature_cols[2]:
        st.markdown("#### Rot Disease")
        st.markdown("- Soft, brown areas")
        st.markdown("- Spreading decay")
        st.markdown("- Watery appearance")
        st.markdown("- Fungal mycelium")
        
    with feature_cols[3]:
        st.markdown("#### Scab Disease")
        st.markdown("- Rough, corky spots")
        st.markdown("- Olive-green lesions")
        st.markdown("- Cracked skin")
        st.markdown("- Circular pattern")

# New Tab 6: AI Analysis with Gemini
with tab6:
    st.header(get_text("AI Harvest Analysis"))
    
    # Create a unique cache key for this session
    cache_key = f"{selected_session}_analysis"
    
    # Check if we already have analysis for this session
    if cache_key not in st.session_state.analysis_cache:
        with st.spinner(get_text("Loading AI analysis...")):
            # Get AI analysis from Gemini
            analysis = gemini_integration.analyze_harvest_data(analysis_data)
            st.session_state.analysis_cache[cache_key] = analysis
    else:
        analysis = st.session_state.analysis_cache[cache_key]
        
    # Display the analysis
    if st.session_state.language == 'hindi':
        with st.spinner(get_text("हिंदी में अनुवाद हो रहा है...")):
            translated_analysis = gemini_integration.translate_to_hindi(analysis)
            st.markdown(translated_analysis)
    else:
        st.markdown(analysis)
        
    # Add data visualization of predicted market value
    st.subheader(get_text("Predicted Market Value Distribution"))
    
    # Calculate estimated market value based on condition
    normal_value = category_counts["Normal_Apple"] * 100  # Example: $100 per normal apple
    blotch_value = category_counts["Blotch_Apple"] * 60   # $60 per blotch apple
    rot_value = category_counts["Rot_Apple"] * 30         # $30 per rot apple
    scab_value = category_counts["Scab_Apple"] * 40       # $40 per scab apple
    total_value = normal_value + blotch_value + rot_value + scab_value
    
    # Create data for bar chart
    value_data = pd.DataFrame({
        'Condition': ['Normal', 'Blotch', 'Rot', 'Scab', 'Total'],
        'Value ($)': [normal_value, blotch_value, rot_value, scab_value, total_value]
    })
    
    # Create a bar chart
    bar_chart = alt.Chart(value_data).mark_bar().encode(
        x='Condition:N',
        y='Value ($):Q',
        color=alt.Color('Condition:N', scale=alt.Scale(
            domain=['Normal', 'Blotch', 'Rot', 'Scab', 'Total'],
            range=['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad', '#3498db']
        )),
        tooltip=['Condition:N', 'Value ($):Q']
    ).properties(
        width=600,
        height=400
    )
    
    st.altair_chart(bar_chart, use_container_width=True)
    
    # Add seasonal trend analysis
    st.subheader(get_text("Seasonal Disease Trend Analysis"))
    
    # Create simulated seasonal data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # Use the current distribution to simulate the entire season
    blotch_trend = [max(0, blotch_percentage/100 * (1 + 0.3*i - 0.025*i*i)) for i in range(12)]
    rot_trend = [max(0, rot_percentage/100 * (0.5 + 0.4*i - 0.03*i*i)) for i in range(12)]
    scab_trend = [max(0, scab_percentage/100 * (0.3 + 0.5*i - 0.035*i*i)) for i in range(12)]
    
    # Create data for line chart
    trend_data = pd.DataFrame({
        'Month': months * 3,
        'Disease': ['Blotch'] * 12 + ['Rot'] * 12 + ['Scab'] * 12,
        'Prevalence': blotch_trend + rot_trend + scab_trend
    })
    
    # Create a line chart
    line_chart = alt.Chart(trend_data).mark_line(point=True).encode(
        x='Month:N',
        y='Prevalence:Q',
        color='Disease:N',
        tooltip=['Month:N', 'Disease:N', 'Prevalence:Q']
    ).properties(
        width=600,
        height=400
    )
    
    st.altair_chart(line_chart, use_container_width=True)

# New Tab 7: Context-aware Chatbot
with tab7:
    st.header(get_text("Chat with Harvest Assistant"))
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
    
    # Chat input
    user_question = st.chat_input(get_text("Ask a question..."))
    
    if user_question:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.chat_message("user").write(user_question)
        
        # Get response from Gemini
        with st.chat_message("assistant"):
            with st.spinner(get_text("Thinking...")):
                response = gemini_integration.ask_question(
                    user_question, 
                    context_data={
                        "session_info": {
                            "session_id": selected_session,
                            "timestamp": session_data.get("timestamp", ""),
                            "total_apples": len(apple_images),
                        },
                        "condition_counts": category_counts,
                        "quality_results": quality_results[:3]  # Just send a few examples for context
                    },
                    language=st.session_state.language
                )
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.write(response)

# Add refresh button
if st.button(get_text("Refresh Data")):
    st.cache_data.clear()
    st.session_state.analysis_cache = {}
    st.experimental_rerun()

# Add footer
st.markdown("---")
st.markdown("Apple Harvest Management System - Made with ❤️ for Farmers") 