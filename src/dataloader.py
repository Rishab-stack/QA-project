import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

def get_generators(train_dir, val_dir):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=None)

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        class_mode="binary",
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


def visualize_grayscale_samples(generator, num_images=8):
    """Display grayscale images from the generator"""
    # Get one batch
    images, labels = next(generator)
    
    # Create subplot grid
    rows = 2
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    
    for i in range(min(num_images, len(images))):
        ax = axes[i // cols, i % cols]
        
        # Remove channel dimension for display (224, 224, 1) -> (224, 224)
        img = images[i].squeeze()
        
        # Display grayscale image
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Label: {'Wear' if labels[i] >= 0.5 else 'Good'}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('grayscale_samples.png')
    plt.show()
    print("✅ Saved as 'grayscale_samples.png'")


# Usage
if __name__ == "__main__":
    train_gen, val_gen = get_generators("data/train", "data/val")
    
    # Visualize training images
    visualize_grayscale_samples(train_gen)
