import os
import sys
import gradio as gr
import tempfile
import base64
import os
import uuid
import cv2
import gradio as gr
import numpy as np
import requests
from typing_extensions import Literal
from openai import OpenAI
from dotenv import load_dotenv


MARKDOWN = """
# WebcamGPT 💬 + 📸

webcamGPT is a tool that allows you to chat with video using OpenAI Vision API.

Visit [awesome-openai-vision-api-experiments](https://github.com/roboflow/awesome-openai-vision-api-experiments) 
repository to find more OpenAI Vision API experiments or contribute your own.
"""
AVATARS = (
    "https://media.roboflow.com/spaces/roboflow_raccoon_full.png",
    "https://media.roboflow.com/spaces/openai-white-logomark.png",
)
IMAGE_CACHE_DIRECTORY = "data"
API_URL = "https://api.openai.com/v1/chat/completions"


def get_openai_key() -> str:
    openai_key = os.getenv("OPENAI_API_KEY")  # Use system environment variable
    if openai_key is None:  # Use environment variable from .env file
        load_dotenv()
        openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key is None:
        print("Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    return openai_key


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


def compose_headers(api_key: str) -> dict:
    return {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}


def prompt_image(api_key: str, image: np.ndarray, prompt: str) -> str:
    headers = compose_headers(api_key=api_key)
    payload = compose_payload(image=image, prompt=prompt)
    response = requests.post(url=API_URL, headers=headers, json=payload).json()

    if "error" in response:
        raise ValueError(response["error"]["message"])
    return response["choices"][0]["message"]["content"]


def cache_image(image: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False) as temp_file:
        image_filename = temp_file.name
        cv2.imwrite(image_filename, image)

    # image_filename = f"{uuid.uuid4()}.jpeg"
    # os.makedirs(IMAGE_CACHE_DIRECTORY, exist_ok=True)
    # image_path = os.path.join(IMAGE_CACHE_DIRECTORY, image_filename)
    # cv2.imwrite(image_path, image)
    return image_filename


def respond(image: np.ndarray, prompt: str, chat_history):
    image = preprocess_image(image=image)
    cached_image_path = cache_image(image)
    response = prompt_image(api_key=openai_key, image=image, prompt=prompt)
    chat_history.append(((cached_image_path,), None))
    chat_history.append((prompt, response))
    return "", chat_history


def generate_image(
    text: str,
    model: Literal["dall-e-2", "dall-e-3"],
    quality: Literal["standard", "hd"],
    size: Literal["1024x1024", "1024x1792", "1792x1024"],
) -> str:
    try:
        client = OpenAI(api_key=openai_key)
        response = client.images.generate(
            prompt=text,
            model=model,
            quality=quality,
            size=size,
            n=1,
        )
    except Exception as error:
        print(str(error))
        raise gr.Error(
            "An error occurred while generating speech. Please check your API key and come back try again."
        )

    return response.data[0].url


def text_to_speech(
    text: str,
    model: Literal["tts-1", "tts-1-hd"],
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    output_file_format: Literal["mp3", "opus", "aac", "flac", ""] = "",
    speed: float = 1.0,
) -> str:
    try:
        client = OpenAI(api_key=openai_key)
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=output_file_format,
            speed=speed,
        )

    except Exception as error:
        print(str(error))
        raise gr.Error(
            "An error occurred while generating speech. Please check your API key and come back try again."
        )

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_file.write(response.content)

    temp_file_path = temp_file.name

    return temp_file_path


def gui():
    with gr.Blocks() as gradio:
        with gr.Tab(label="Chat with Vision"):
            # gr.Markdown("# <center> Chat </center>")
            with gr.Column():
                gr.Markdown(MARKDOWN)
                with gr.Row():
                    vision_on = gr.Checkbox(label="Vision On", default=True)
                    tts_on = gr.Checkbox(label="Text to Speech On", default=True)
                with gr.Row():
                    webcam = gr.Image(source="webcam", streaming=True)
                    with gr.Column():
                        chatbot = gr.Chatbot(
                            height=500, bubble_full_width=False, avatar_images=AVATARS
                        )
                        message_textbox = gr.Textbox()
                        clear_button = gr.ClearButton([message_textbox, chatbot])

            message_textbox.submit(
                fn=respond,
                inputs=[webcam, message_textbox, chatbot],
                outputs=[message_textbox, chatbot],
            )

        with gr.Tab(label="Text to Speech"):
            gr.Markdown("# <center> Text to Speech </center>")
            with gr.Row(variant="panel"):
                model = gr.Dropdown(
                    choices=["tts-1", "tts-1-hd"], label="Model", value="tts-1"
                )
                voice = gr.Dropdown(
                    choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                    label="Voice Options",
                    value="alloy",
                )
                output_file_format = gr.Dropdown(
                    choices=["mp3", "opus", "aac", "flac"],
                    label="Output Options",
                    value="mp3",
                )
                speed = gr.Slider(
                    minimum=0.25, maximum=4.0, value=1.0, step=0.01, label="Speed"
                )

            text = gr.Textbox(
                label="Input text",
                placeholder='Enter your text and then click on the "Text-To-Speech" button, '
                "or simply press the Enter key.",
            )
            btn = gr.Button("Text-To-Speech")
            output_audio = gr.Audio(label="Speech Output", autoplay=True)

            text.submit(
                fn=text_to_speech,
                inputs=[text, model, voice, output_file_format, speed],
                outputs=output_audio,
                api_name="text_to_speech",
            )
            btn.click(
                fn=text_to_speech,
                inputs=[text, model, voice, output_file_format, speed],
                outputs=output_audio,
                api_name=False,
            )

        with gr.Tab(label="Image Generation"):
            gr.Markdown("# <center> Image Generation </center>")
            with gr.Row(variant="panel"):
                model = gr.Dropdown(
                    choices=["dall-e-2", "dall-e-3"], label="Model", value="dall-e-3"
                )
                quality = gr.Dropdown(
                    choices=["standard", "hd"], label="Quality", value="standard"
                )
                size = gr.Dropdown(
                    choices=["1024x1024", "1792x1024", "1024x1792"],
                    label="Size",
                    value="1024x1024",
                )

            text = gr.Textbox(
                label="Input Text",
                placeholder='Enter your text and then click on the "Image Generate" button, '
                "or simply press the Enter key.",
            )
            btn = gr.Button("Image Generate")
            output_image = gr.Image(label="Image Output")

            text.submit(
                fn=generate_image,
                inputs=[text, model, quality, size],
                outputs=output_image,
                api_name="generate_image",
            )
            btn.click(
                fn=generate_image,
                inputs=[text, model, quality, size],
                outputs=output_image,
                api_name=False,
            )

    gradio.launch()


if __name__ == "__main__":
    openai_key = get_openai_key()  # global api key
    gui()
