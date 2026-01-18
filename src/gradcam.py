import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array
import cv2

IMG_SIZE = (224, 224)

def get_img_array(img_path):
    """Load and preprocess image"""
    img = load_img(img_path, color_mode="grayscale", target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="last_conv"):
    """Generate activation-based heatmap"""
    # Call model once to build the graph
    _ = model(img_array)
    
    # Get last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    
    # Build intermediate model
    intermediate_model = tf.keras.Model(
        inputs=model.inputs[0],
        outputs=last_conv_layer.output
    )
    
    # Get activations
    activations = intermediate_model.predict(img_array, verbose=0)
    
    # Average across all filters
    heatmap = np.mean(activations[0], axis=-1)
    
    # Normalize
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    
    # Resize to original image size
    heatmap = cv2.resize(heatmap, (IMG_SIZE[0], IMG_SIZE[1]))
    
    return heatmap


def display_gradcam(img_path, heatmap, cam_path="gradcam_output.jpg", alpha=0.4):
    """Display and save Grad-CAM overlay"""
    # Load original image
    img = load_img(img_path, color_mode="grayscale", target_size=IMG_SIZE)
    img = img_to_array(img).astype("uint8")

    # Rescale heatmap to 0-255
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Apply jet colormap
    jet_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)

    # Convert grayscale to RGB
    img_rgb = np.stack([img[:,:,0], img[:,:,0], img[:,:,0]], axis=-1)

    # Superimpose
    superimposed_img = jet_heatmap * alpha + img_rgb * (1 - alpha)
    superimposed_img = np.uint8(superimposed_img)

    # Save
    cv2.imwrite(cam_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    # Display
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img[:,:,0], cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.title("Activation Heatmap")
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.title("Wear Hotspots")
    plt.imshow(superimposed_img)
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Saved heatmap to {cam_path}")

if __name__ == "__main__":
    # Load model
    model = tf.keras.models.load_model("wear_classifier.h5")
    
    # Test image
    img_path = "data/val/sharpen/3.jpg"
    
    # Prepare image
    img_array = get_img_array(img_path)
    
    # Prediction
    preds = model.predict(img_array)
    print(f"Prediction: {'Wear (mild)' if preds[0][0] >= 0.5 else 'Good'} - Confidence: {preds[0][0]:.3f}")
    
    # Generate heatmap (activation-based, works without gradients)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="last_conv")
    
    # Display
    display_gradcam(img_path, heatmap, cam_path="gradcam_output.jpg")
