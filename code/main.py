# System and logging configuration
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Import the required libraries
import cv2
import numpy as np
import requests
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import base64


# Preprocess image for VGG16 model input
def preprocess_image(image_path):
    img = cv2.imread(image_path)  
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img_resized = cv2.resize(img_rgb, (224, 224)) 
    img_array = image.img_to_array(img_resized) 
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = preprocess_input(img_array) 
    return img_array

# Extract features using pre-trained VGG16 model
def extract_features(img_array):
    model = VGG16(weights='imagenet', include_top=False)  
    features = model.predict(img_array)  
    return features

# Detect font using WhatFontIs API
def detect_font_using_api(image_path, api_key):
    url = "https://www.whatfontis.com/api2/"
    img_base64 = encode_image_to_base64(image_path)  

    payload = {
        "API_KEY": api_key,
        "IMAGEBASE64": "1",
        "urlimagebase64": img_base64,
        "NOTTEXTBOXSDETECTION": "1", 
        "FREEFONTS": "0",
        "limit": 1 
    }

    response = requests.post(url, data=payload)  

    if response.status_code == 200:
        fonts = response.json()  
        if fonts:
            print(f"Detected Font: {fonts[0]['title']}")  
        else:
            print("No fonts detected.")
    else:
        print(f"Error: {response.status_code}")

# Encode image to base64 format
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Example usage
image_path = r"C:\Users\krish\OneDrive\Desktop\image.jpg"  # Path to image
api_key = "01e24a3fb455a8737d78e05b2558ab7771da974c1a71b1f893e1eb270a7c34ee"  # API key

img_array = preprocess_image(image_path)  # Preprocess image
features = extract_features(img_array)  # Extract features (optional)
detect_font_using_api(image_path, api_key)  # Detect font using API