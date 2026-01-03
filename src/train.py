import os
from dataloader import get_generators
from model import build_model

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
VAL_DIR   = os.path.join(BASE_DIR, "data", "val")

def main():
    train_gen, val_gen = get_generators(TRAIN_DIR, VAL_DIR)
    model = build_model()

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10
    )

    model.save(os.path.join(BASE_DIR, "wear_classifier.h5"))

if __name__ == "__main__":
    main()
