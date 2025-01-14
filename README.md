<h1 align="center">ColourPulse</h1>
The code performs automatic colorization of black-and-white images using a pre-trained deep learning model. It loads the model and cluster centers, processes the input image, predicts color channels, combines them with the luminance channel, and converts the result to a colorized image.

## Execution Guide:
1. Run the following command line in the terminal:
   ```
   pip install opencv-python opencv-contrib-python numpy
   ```

2. Enter the path of the black and white images in the code, also enter the output directory.

3. Download the following models and paste their path in the code:

   a. [colorization_deploy_v2.prototxt](https://github.com/kr1shnasomani/ColourPulse/blob/main/model/colorization_deploy_v2.prototxt)

   b. [colorization_release_v2.caffemodel](https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1)

   c. [pts_in_hull.npy](https://github.com/kr1shnasomani/ColourPulse/blob/main/model/pts_in_hull.npy)

4. Run the code and it will output a colour image

## Model Prediction:

Image Input:

![blackwhite](https://github.com/user-attachments/assets/6cb433dc-426e-44f3-8b42-f15f9e4a69a5)

Image Output:

![colour](https://github.com/user-attachments/assets/e6c9acc0-9ed0-478c-95a3-0435bfe7355a)

## Overview:
This script implements **automatic colorization of black-and-white images** using a pre-trained deep learning model in OpenCV. Below is a step-by-step breakdown:

#### 1. **Library Imports**
   - The script uses essential libraries: 
     - `numpy` for numerical computations.
     - `cv2` (OpenCV) for image processing.
     - `os` for file path management.

#### 2. **Model Paths**
   - Specifies the paths to the required model files:
     - `prototxt`: Defines the network architecture.
     - `caffemodel`: Contains the pre-trained weights.
     - `npy`: Stores cluster centers for color distribution.
   - Ensures paths are dynamically adjusted and checks the existence of critical files.

#### 3. **Model Initialization**
   - Loads the pre-trained model (`.prototxt` and `.caffemodel`) using OpenCV's `cv2.dnn.readNetFromCaffe`.
   - Loads the color cluster centers from the `.npy` file.
   - Modifies the network by adding cluster centers as 1x1 convolutions.

#### 4. **Colorization Function (`colorize_image`)**
   - **Input**: Path to a black-and-white image.
   - **Steps**:
     1. Reads the input image and verifies its existence.
     2. Converts the image to the **LAB color space**, where:
        - `L`: Lightness (input channel for colorization).
        - `a` and `b`: Color channels (predicted by the model).
     3. Preprocesses the image:
        - Normalizes pixel values.
        - Resizes the image to match the model input dimensions.
        - Extracts and adjusts the `L` channel.
     4. Feeds the processed `L` channel to the network.
     5. Predicts the `a` and `b` channels and resizes them to the original image dimensions.
     6. Combines the original `L` channel with the predicted `a` and `b` channels.
     7. Converts the LAB image back to the **BGR color space** for display.
     8. Clamps pixel values to the valid range and converts the image to 8-bit format.

### Key Features:
- Utilizes OpenCV's **DNN module** to load and process pre-trained deep learning models.
- Automatically converts and colorizes black-and-white images using the **LAB color space**.
- Provides robust error handling for missing files or invalid inputs.
