import os
import tempfile
import base64
import uuid
import requests

import cv2
import numpy as np
from typing_extensions import Literal
from openai import OpenAI

from openai_client import openai_key
from utils import *


def preprocess_image(image: np.ndarray) -> np.ndarray:
    image = np.fliplr(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def encode_image_to_base64(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Could not encode image to JPEG format.")

    encoded_image = base64.b64encode(buffer).decode("utf-8")
    return encoded_image


def compose_payload(image: np.ndarray, prompt: str) -> dict:
    base64_image = encode_image_to_base64(image)
    return {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }


def prompt_image(api_key: str, image: np.ndarray, prompt: str) -> str:
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = compose_payload(image=image, prompt=prompt)
    response = requests.post(url=api_url, headers=headers, json=payload).json()

    if "error" in response:
        raise ValueError(response["error"]["message"])
    return response["choices"][0]["message"]["content"]


def respond_with_vision(image: np.ndarray, prompt: str, chat_history):
    image = preprocess_image(image=image)
    image_path = cache_image(image)
    response = prompt_image(api_key=openai_key, image=image, prompt=prompt)
    chat_history.append(((image_path, None), None))
    chat_history.append((prompt, response))
    return "", chat_history
