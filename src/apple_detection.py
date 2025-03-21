import google.generativeai as genai
import random

from dotenv import load_dotenv
import PIL.Image

load_dotenv()

# REPLACE WITH YOUR API KEY
keys = "<YOUR_API_KEY>"
genai.configure(api_key=keys)

def classify_apples(image_path):
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    }

    model = genai.GenerativeModel(model_name = "gemini-1.5-flash", generation_config=generation_config)

    sample_file_1 = PIL.Image.open(image_path)


    prompt = "You are a agriculture expert.Comment on the given image as an expert, keep it short. REPLY IN HINDI"

    response = model.generate_content([sample_file_1, prompt])

    return response.text

def get_fertilizers(disease):
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }

    model = genai.GenerativeModel(model_name = "gemini-1.5-flash", generation_config=generation_config)

    prompt = f"You are a agriculture expert. Given the disease:{disease} on apple tree suggest some fertilizers. Keep it short and only return fertilizers names"

    response = model.generate_content([prompt])

    return response.text

def get_qwt(readings):
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }

    model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)

    prompt = f"You are a agriculture expert. Given the soil and temperature conditions:{readings} for an apple tree give some suggestion. Keep it short and descriptive"

    response = model.generate_content([prompt])

    return response.text