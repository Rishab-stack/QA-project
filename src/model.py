from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

INPUT_SHAPE = (224, 224, 1)

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE, name="conv1"),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu", name="conv2"),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu", name="last_conv"),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")  # binary output
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model
