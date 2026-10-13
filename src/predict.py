"""Inference helpers for the CIFAR-10 CNN demo app."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
from tensorflow import keras

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
IMAGE_SIZE = (32, 32)


def load_or_create_demo_model(model_path: str) -> keras.Model:
    """Load a trained Keras model from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. "
            "Train the model in classification_cnn.ipynb or set MODEL_PATH."
        )
    return keras.models.load_model(path)


def preprocess(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a normalized (1, 32, 32, 3) batch."""
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype="float32") / 255.0
    return np.expand_dims(array, axis=0)


def predict_image(model: keras.Model, image: Image.Image) -> Dict[str, float]:
    """Return class probabilities for a single image."""
    if image is None:
        return {}
    probs = model.predict(preprocess(image), verbose=0)[0]
    return {name: float(p) for name, p in zip(CLASS_NAMES, probs)}
