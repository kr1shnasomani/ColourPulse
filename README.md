<h1 align="center">FontFinder</h1>
The script preprocesses an image, extracts features using VGG16 and detects fonts via the WhatFontIs API. It includes image-to-base64 encoding and securely requires an API key for font identification.

## Execution Guide:
1. Run the following command line in the terminal:
   ```
   pip install tensorflow opencv-python numpy requests keras-preprocessing
   ```

2. Go to the site [What Font Is](https://www.whatfontis.com/API-identify-fonts-from-image.html) > create an account > copy the API key > paste it in the code

3. Copy paste the path of the image for whose font you want to detect

4. Run the code and it will output its prediction

## Model Prediction:

Image Input:

![image](https://github.com/user-attachments/assets/089385f4-1e58-428e-a93b-99cf9e5babbb)

Output:

`Detected Font: Bernard MT Condensed`

## Overview:
The Python code is a multi-functional script that combines image processing, feature extraction using a pre-trained deep learning model, and font detection via an API. Here's an overview of its key components:

### **1. System and Logging Configuration**
- Sets TensorFlow logging to minimal verbosity to suppress unnecessary warnings (`os.environ['TF_CPP_MIN_LOG_LEVEL']`).
- Configures logging to suppress TensorFlow and OpenCV debug information for cleaner output.

### **2. Libraries and Dependencies**
- **OpenCV (`cv2`)**: For image processing tasks such as reading, resizing, and color conversion.
- **NumPy**: For numerical operations on image arrays.
- **TensorFlow & Keras**: To use the pre-trained VGG16 model for feature extraction.
- **Requests**: To interact with the WhatFontIs API.
- **Base64**: To encode images for API submission.

### **3. Functions**
**a. preprocess_image:** Prepares an image for input into the VGG16 model:
  1. Reads the image from the specified path.
  2. Converts the image from BGR (OpenCV format) to RGB.
  3. Resizes the image to 224x224 (required input size for VGG16).
  4. Converts the image to an array and preprocesses it using `preprocess_input` (scales pixel values).

**b. extract_features:** Uses the pre-trained VGG16 model to extract deep features:
  1. Loads the VGG16 model with pre-trained weights (ImageNet) and excludes the top classification layer (`include_top=False`).
  2. Runs a forward pass on the preprocessed image to generate feature maps.

**c. detect_font_using_api:** Detects the font in an image using the **WhatFontIs API**:
  1. Encodes the image into base64 format.
  2. Prepares the payload with API key, base64 image, and other configurations.
  3. Sends a POST request to the API endpoint.
  4. Parses the response to print the detected font name or error details.

**d. encode_image_to_base64:** Converts the input image into base64 format for API compatibility:
  1. Reads the image as binary data.
  2. Encodes the binary data into a base64 string.

### **Use Case**
- **Font Detection**: Identify fonts used in an image using a combination of image preprocessing and an external API.
- **Feature Extraction**: Generate deep features from images for further analysis (e.g., image classification, clustering).

### **Key Points**
- The **VGG16 model** is used only for feature extraction and does not directly contribute to font detection.
- The **WhatFontIs API** handles the actual font detection process, making this script an integration of local image processing and remote font recognition.
- **Error Handling**: Basic error handling is implemented for the API response.

This script is suitable for tasks involving image analysis and font identification in a streamlined workflow.
