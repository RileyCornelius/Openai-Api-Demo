import base64
import requests

import cv2
import numpy as np
from typing_extensions import Literal
from openai import OpenAI
from utils import *
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


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


def chat_with_image(client: OpenAI, prompt: str, image: np.ndarray) -> str:
    params = compose_payload(image, prompt)
    response = client.chat.completions.create(**params)
    response.choices[0].message.content

    if "error" in response:
        raise ValueError(response["error"]["message"])
    return response.choices[0].message.content


def chat_with_image_stream(client: OpenAI, prompt: str, image: np.ndarray):
    params = compose_payload(image, prompt)
    stream = client.chat.completions.create(**params, stream=True)

    if stream.response.is_error:
        raise ValueError(stream.response.status_code, stream.response.reason_phrase)
    return stream
