"""Gradio demo app for CIFAR-10 image classification."""

from __future__ import annotations

import os
from typing import Dict

import gradio as gr
from PIL import Image

from src.predict import load_or_create_demo_model, predict_image

MODEL_PATH = os.getenv("MODEL_PATH", "models/baseline_model.keras")
model = load_or_create_demo_model(MODEL_PATH)


def classify(image: Image.Image) -> Dict[str, float]:
    """Return class probabilities for the uploaded image."""
    return predict_image(model, image)


description = """
Upload an image. The model resizes it to 32x32 and classifies it into one of the 10 CIFAR-10 classes.

Note: the model is trained on CIFAR-10, so it works best on images similar to the dataset classes:
airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships and trucks.
"""

with gr.Blocks(title="CNN CIFAR-10 Demo") as demo:
    gr.Markdown("# CNN Image Classification — Gradio Demo")
    gr.Markdown(description)

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload an image")
        label_output = gr.Label(num_top_classes=5, label="Model predictions")

    predict_button = gr.Button("Classify")
    predict_button.click(fn=classify, inputs=image_input, outputs=label_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
