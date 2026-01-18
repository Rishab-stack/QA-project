import os
import tensorflow as tf
from gradcam import display_gradcam

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "wear_classifier.h5")

model = tf.keras.models.load_model(MODEL_PATH)

# optional debug
print("Layers:", [layer.name for layer in model.layers])

if __name__ == "__main__":
    test_img = os.path.join(BASE_DIR, "data", "val", "sharpen", "3.jpg")  # adjust path
    display_gradcam(test_img, model, last_conv_layer_name="last_conv")
