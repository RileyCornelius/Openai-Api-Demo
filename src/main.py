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


def get_openai_key() -> str:
    openai_key = os.getenv("OPENAI_API_KEY")  # Use system environment variable
    if openai_key is None:  # Use environment variable from .env file
        load_dotenv()
        openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key is None:
        print("Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    return openai_key


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
