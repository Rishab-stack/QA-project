import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

IMG_SIZE = (224, 224)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "wear_classifier.h5")

model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(img_path):
    img = image.load_img(img_path, color_mode="grayscale", target_size=IMG_SIZE)
    x = image.img_to_array(img)           # (224, 224, 1)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)         # (1, 224, 224, 1)

    prob = model.predict(x)[0][0]
    label = "wear (mild)" if prob >= 0.5 else "no wear (Good)"
    return label, float(prob)

if __name__ == "__main__":
    test_path = "path/to/new_image.jpg"
    label, prob = predict_image(test_path)
    print(f"Prediction: {label}, probability={prob:.3f}")
