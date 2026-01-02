import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

def get_generators(train_dir, val_dir):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=None)

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        color_mode="grayscale",   # converts to 1 channel
        class_mode="binary",      # mild vs Good
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        class_mode="binary",
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Class indices:", train_gen.class_indices)
    return train_gen, val_gen
