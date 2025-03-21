import os
from google.cloud import storage
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import cv2
import math
import tensorflow as tf
from imageai.Classification.Custom.training_params import model
from tensorflow.keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.image as mpimg
from uuid import uuid4

from pymongo import MongoClient

from apple_detection import classify_apples, get_fertilizers, get_qwt

load_dotenv()

apple_disease = {
    0: 'complex',
    1: 'frog_eye_leaf_spot',
    2: 'frog_eye_leaf_spot complex',
    3: 'healthy',
    4: 'powdery_mildew',
    5: 'powdery_mildew complex',
    6: 'rust',
    7: 'rust complex',
    8: 'rust frog_eye_leaf_spot',
    9: 'scab',
    10: 'scab frog_eye_leaf_spot',
    11: 'scab frog_eye_leaf_spot complex'
}

@st.cache_resource
def run_initial_code():
    return "initialized"


def download_images_from_bucket(bucket_name, download_path, prefix="uploads/"):
    """
    Download images from a Google Cloud Storage bucket to a local folder, preserving the folder structure.

    :param bucket_name: Name of the GCS bucket
    :param download_path: Local folder to store the downloaded images
    :param prefix: Prefix for the files to filter (e.g., "uploads/")
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)  # List all blobs with the specified prefix

    for blob in blobs:
        # Define local folder structure based on the GCS path
        local_folder_path = os.path.join(download_path, *blob.name.split('/')[:-1])  # Exclude the file name
        os.makedirs(local_folder_path, exist_ok=True)

        # Define local file path
        local_file_path = os.path.join(local_folder_path, blob.name.split('/')[-1])

        # Download the blob to the local file path
        blob.download_to_filename(local_file_path)
        print(f"Downloaded {blob.name} to {local_file_path}")

if  not os.path.isdir("downloaded_apples/uploads"):
    bucket_name = "apple_2004"
    download_path = "downloaded_apples"
    download_images_from_bucket(bucket_name, download_path)

#model_leafs = load_model('models/apple_leaf.h5')
model_apples = load_model('models/apple_quality_model.h5')

# Function to load images from subdirectories

def list_image_paths(root_directory):
    """
    Recursively lists the paths of all image files in the root directory and its subdirectories.

    Args:
        root_directory (str): Path to the root directory.

    Returns:
        List[str]: A list of file paths for all images in the directory tree.
    """
    image_paths = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')  # Add more extensions if needed

    for root, _, files in os.walk(root_directory):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, file))

    return image_paths

# Streamlit app logic
st.title("Apple Orchards")

# Directory path input
directory = "./downloaded_apples"

images = list_image_paths(directory)

class_labels = ['Blotch_Apple', 'Healthy', 'Rot_Apple', 'Scab_Apple']
predicted_images = []
for img_path in images:
    img = Image.open(img_path)
    target_size = (150, 150)
    img = img.resize(target_size, Image.LANCZOS)

    # Optional: Enhance contrast and sharpness if needed
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)  # Increase contrast
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)  # Sharpen the image

    img_array = np.array(img)

    img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)

    # Normalize and expand the image dimensions for prediction
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Make prediction
    predictions = model_apples.predict(img_array)
    predicted_class_index = np.argmax(predictions)  # Get the index of the highest probability
    predicted_class = class_labels[predicted_class_index]

    predicted_images.append(predicted_class)

n_images = len(images)
n_cols = math.ceil(math.sqrt(n_images))  # Number of columns
n_rows = math.ceil(n_images / n_cols)    # Number of rows

# Create the figure and axes
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
axes = axes.ravel()  # Flatten the 2D array of axes into 1D for easier iteration

# Plot each image in the grid
for i, (img_path, label) in enumerate(zip(images, predicted_images)):
    img = mpimg.imread(img_path)  # Read the image
    axes[i].imshow(img)          # Display the image
    axes[i].set_title(label)     # Set the title
    axes[i].axis('off')          # Turn off the axis

# Remove unused axes (if any)
for j in range(len(images), len(axes)):
    axes[j].axis('off')

id = str(uuid4())
fig_path = f'{id}.png'
plt.savefig(fig_path)
st.pyplot(fig)

# Gemini logic
response = classify_apples(fig_path)

st.write(response)

# st.header("Upload Leaf Image for Analysis")
# uploaded_file = st.file_uploader("Upload an image of lead (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])
#
# if uploaded_file is not None:
#     img = load_img(uploaded_file, target_size=(128, 128))
#     # Convert the image to a NumPy array
#     st.image(uploaded_file)
#     img_array = img_to_array(img)
#     # Add an extra dimension for batch size (model expects a batch of images)
#     img_array = np.expand_dims(img_array, axis=0)
#     # Normalize the image if necessary (depending on how your model was trained)
#     # For example, if your model was trained with pixel values in the range [0, 1], divide by 255
#     img_array /= 255.0
#     model = load_model('models/apple_leaf.h5')
#     # Predict with the model
#     preds = model_leafs.predict(img_array)
#
#     # Print predictions
#     leaf_class = np.argmax(preds)
#
#     st.write(f"Disease : {apple_disease.get(leaf_class)}")
#
#     response_leaf = get_fertilizers(apple_disease.get(leaf_class))
#     st.write(f"Suggested Fertilizers: {response_leaf}")


st.title("Soil Analysis")
client = MongoClient("mongodb+srv://shreysalunke1:Zd6h3AHrrPuIqLgM@cluster0.mf0oh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.agriculture
collection = db.soil_data

# Assuming there is a 'timestamp' field that records the document's creation time
latest_document = collection.find().sort("created_at", -1).limit(1)
print(latest_document)
# Print the latest document
for doc in latest_document:
    response_qwt = get_qwt(doc)
    st.write(response_qwt)
