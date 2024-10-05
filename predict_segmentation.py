import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the model
model = load_model('unet_model.h5')

# Preprocess the input image and predict the mask
def predict_tumour(image_path, model):
    img = load_img(image_path, target_size=(256, 256), color_mode='grayscale')
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict the mask
    pred_mask = model.predict(img)
    pred_mask = np.squeeze(pred_mask)  # Remove extra dimensions

    print("Unique values in raw predicted mask:", np.unique(pred_mask))  # Debugging

    # Lower the threshold to account for low values
    pred_mask = (pred_mask > 0.05).astype(np.float32)
    print("Unique values in thresholded mask:", np.unique(pred_mask))  # Debugging

    return pred_mask

# Visualize the MRI and predicted mask
def visualize_prediction(image_path, pred_mask):
    img = load_img(image_path, target_size=(256, 256), color_mode='grayscale')
    img = img_to_array(img) / 255.0

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.squeeze(img), cmap='gray')
    plt.title('MRI Image')

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
    plt.title('Predicted Tumour Mask')

    plt.savefig('predicted_mask_output.png')
    plt.show()