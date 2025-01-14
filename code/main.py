# Import the required libraries
import numpy as np
import cv2
import os

# Paths to model files
prototxt = r"C:\Users\krish\OneDrive\Desktop\Projects\ColorPulse\model\colorization_deploy_v2.prototxt"
caffemodel = r"C:\Users\krish\OneDrive\Desktop\Projects\ColorPulse\model\colorization_release_v2.caffemodel"
npy = r"C:\Users\krish\OneDrive\Desktop\Projects\ColorPulse\model\pts_in_hull.npy"

# Ensure correct paths
prototxt = os.path.join(os.path.dirname(__file__), prototxt)
caffemodel = os.path.join(os.path.dirname(__file__), caffemodel)
npy = os.path.join(os.path.dirname(__file__), npy)

if not os.path.isfile(caffemodel):
    raise FileNotFoundError("Missing model file 'colorization_release_v2.caffemodel'")

# Load model and cluster centers
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
pts = np.load(npy)

# Add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorize_image(image_path):
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    # Normalize and convert image to LAB color space
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # Resize the LAB image and extract the L channel
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Predict the 'a' and 'b' channels
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize the predicted 'ab' channels to match the input image
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    # Combine the original L channel with the predicted 'ab' channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # Convert LAB image back to BGR color space
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # Convert to 8-bit image
    colorized = (255 * colorized).astype("uint8")
    return colorized

# Example usage
if __name__ == "__main__":
    input_path = r"C:\Users\krish\OneDrive\Desktop\Projects\ColorPulse\dataset\blackwhite.jpg"
    output_path = r"C:\Users\krish\OneDrive\Desktop\Projects\ColorPulse\output\colour.jpg"

    try:
        colorized_image = colorize_image(input_path)
        cv2.imwrite(output_path, colorized_image)
        print(f"Colorized image saved to '{output_path}'")
    except Exception as e:
        print(f"Error: {e}")