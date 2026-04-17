"""Gradio-приложение для демонстрации классификации изображений моделью CIFAR-10."""

from __future__ import annotations

import os
from typing import Dict

import gradio as gr
from PIL import Image

from src.predict import load_or_create_demo_model, predict_image

MODEL_PATH = os.getenv("MODEL_PATH", "models/baseline_model.keras")
model = load_or_create_demo_model(MODEL_PATH)


def classify(image: Image.Image) -> Dict[str, float]:
    """Возвращает вероятности классов для загруженного изображения."""
    return predict_image(model, image)


description = """
Загрузите изображение. Модель приведет его к размеру 32x32 и выполнит классификацию по 10 классам CIFAR-10.

Важно: модель обучена на CIFAR-10, поэтому лучше всего она работает на изображениях, похожих по тематике на классы датасета:
самолеты, автомобили, птицы, кошки, олени, собаки, лягушки, лошади, корабли и грузовики.
"""

with gr.Blocks(title="CNN CIFAR-10 Demo") as demo:
    gr.Markdown("# CNN классификация изображений — Gradio demo")
    gr.Markdown(description)

    with gr.Row():
        image_input = gr.Image(type="pil", label="Загрузите изображение")
        label_output = gr.Label(num_top_classes=5, label="Предсказания модели")

    predict_button = gr.Button("Распознать")
    predict_button.click(fn=classify, inputs=image_input, outputs=label_output)

    gr.Examples(
        examples=[],
        inputs=image_input,
        outputs=label_output,
        fn=classify,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
