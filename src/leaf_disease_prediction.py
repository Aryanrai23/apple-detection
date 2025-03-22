import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import traceback

class LeafDiseasePredictor:
    def __init__(self, model_path=None):
        """Initialize the leaf disease predictor with a model path."""
        # Define multiple possible paths for the model
        possible_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'leaf_model.h5'),
            "/app/src/models/leaf_model.h5",  # Absolute path in Docker container
            "src/models/leaf_model.h5",       # Relative path
            "./src/models/leaf_model.h5",     # Another relative path
        ]
        
        # If a specific path is provided, try it first
        if model_path is not None:
            possible_paths.insert(0, model_path)
            
        # Try each path until one works
        for path in possible_paths:
            print(f"Attempting to load leaf disease model from: {path}")
            try:
                if os.path.exists(path):
                    print(f"File exists at {path}")
                    # Explicitly specify the TensorFlow version to avoid compatibility issues
                    self.model = load_model(path, compile=False)
                    print(f"Model loaded successfully from {path}")
                    
                    # Get model input shape to verify expectations
                    input_shape = self.model.input_shape
                    output_shape = self.model.output_shape
                    print(f"Model input shape: {input_shape}, output shape: {output_shape}")
                    
                    # The disease classes that this model predicts
                    # Make sure this matches the actual output layer of your model
                    self.classes = ['Apple_cedar_rust', 'Apple_black_rot', 'Apple_scab', 'Healthy']
                    print(f"Classes: {self.classes}")
                    
                    if output_shape[1] != len(self.classes):
                        print(f"WARNING: Model output shape {output_shape[1]} doesn't match the number of classes {len(self.classes)}")
                    
                    # Size that the model expects (from the input shape)
                    if input_shape and len(input_shape) == 4:  # [batch_size, height, width, channels]
                        self.img_size = (input_shape[1], input_shape[2])
                        print(f"Using model's input dimensions: {self.img_size}")
                    else:
                        self.img_size = (224, 224)  # Fallback to standard size
                        print(f"Using default dimensions: {self.img_size}")
                    
                    # Successfully loaded, so exit the loop
                    return
                else:
                    print(f"File does not exist at {path}")
            except Exception as e:
                print(f"Error loading model from {path}: {str(e)}")
                traceback.print_exc()
        
        # If we get here, all paths failed
        print("All model loading attempts failed. Using dummy model.")
        # Create a dummy model for debugging
        self.model = None
        self.classes = ['Apple_cedar_rust', 'Apple_black_rot', 'Apple_scab', 'Healthy']
        self.img_size = (224, 224)
    
    def preprocess_image(self, img):
        """Preprocess the image for the model."""
        try:
            if isinstance(img, bytes):
                img = Image.open(io.BytesIO(img))
                print("Converted bytes to PIL Image")
            
            if isinstance(img, Image.Image):
                print(f"Original image size: {img.size}, mode: {img.mode}")
                
                # Convert to RGB if not already
                if img.mode != "RGB":
                    img = img.convert("RGB")
                    print(f"Converted image to RGB mode")
                
                # Resize the image to the expected input size
                img = img.resize(self.img_size)
                print(f"Resized image to {self.img_size}")
                
                # Convert PIL image to numpy array
                img_array = image.img_to_array(img)
                print(f"Converted to numpy array, shape: {img_array.shape}")
            else:
                # If it's already a numpy array, resize it
                print(f"Input is already a numpy array, shape: {img.shape}")
                img_array = tf.image.resize(img, self.img_size).numpy()
                print(f"Resized array to {img_array.shape}")
            
            # Expand dimensions to create batch of size 1
            img_array = np.expand_dims(img_array, axis=0)
            print(f"Added batch dimension, shape: {img_array.shape}")
            
            # Normalize the image (scale pixel values to [0, 1])
            img_array = img_array / 255.0
            print(f"Normalized pixel values to range [0,1]")
            
            return img_array
        except Exception as e:
            print(f"Error in image preprocessing: {str(e)}")
            traceback.print_exc()
            # Return a dummy array for debugging
            return np.zeros((1, self.img_size[0], self.img_size[1], 3))
    
    def predict(self, img):
        """Predict the leaf disease class from an image."""
        try:
            if self.model is None:
                print("Model not loaded properly, returning dummy prediction")
                return {
                    "predicted_class": "unknown",
                    "confidence": 0.0,
                    "class_probabilities": {cls: 0.25 for cls in self.classes}
                }
                
            # Preprocess the image
            processed_img = self.preprocess_image(img)
            print(f"Preprocessed image shape: {processed_img.shape}")
            
            # Get prediction
            print("Running prediction...")
            predictions = self.model.predict(processed_img, verbose=1)
            print(f"Prediction results shape: {predictions.shape}")
            print(f"Raw predictions: {predictions[0]}")
            
            # Get the class with highest probability
            class_idx = np.argmax(predictions[0])
            print(f"Predicted class index: {class_idx}")
            
            if class_idx < len(self.classes):
                class_name = self.classes[class_idx]
            else:
                print(f"WARNING: Class index {class_idx} is out of range for classes {self.classes}")
                class_name = "unknown"
                
            confidence = float(predictions[0][class_idx])
            print(f"Predicted class: {class_name}, confidence: {confidence}")
            
            # Get all class probabilities
            class_probabilities = {}
            for i in range(min(len(self.classes), len(predictions[0]))):
                class_probabilities[self.classes[i]] = float(predictions[0][i])
            
            print(f"Class probabilities: {class_probabilities}")
            
            return {
                "predicted_class": class_name,
                "confidence": confidence,
                "class_probabilities": class_probabilities
            }
        except Exception as e:
            print(f"Error predicting leaf disease: {str(e)}")
            traceback.print_exc()
            
            # Return a default response instead of failing
            return {
                "error": str(e),
                "predicted_class": "unknown",
                "confidence": 0.0,
                "class_probabilities": {cls: 0.0 for cls in self.classes}
            }
    
    def get_model_summary(self):
        """Get a string representation of the model's architecture."""
        if self.model is None:
            return "Model not loaded properly"
            
        try:
            # Redirect summary to a string
            string_io = io.StringIO()
            self.model.summary(print_fn=lambda x: string_io.write(x + '\n'))
            model_summary = string_io.getvalue()
            string_io.close()
            return model_summary
        except Exception as e:
            print(f"Error getting model summary: {str(e)}")
            return f"Error: {str(e)}"


# Function to get a predictor instance - used to ensure we only load the model once
_predictor_instance = None

def get_predictor():
    """Get a singleton instance of the LeafDiseasePredictor."""
    global _predictor_instance
    
    if _predictor_instance is None:
        print("Creating new LeafDiseasePredictor instance")
        _predictor_instance = LeafDiseasePredictor()
    else:
        print("Reusing existing LeafDiseasePredictor instance")
    
    return _predictor_instance 