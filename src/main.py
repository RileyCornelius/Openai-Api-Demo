import os
import sys
import time
import tempfile
import base64
import os
import uuid
import requests
from typing_extensions import Literal

import cv2
import gradio as gr
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


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


def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


def prompt(text: str) -> str:
    BAXTER_ASSISTANT_ID = "asst_IO7ySDCqFNGz7yuZrov8WLsX"
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=text)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=BAXTER_ASSISTANT_ID)
    wait_on_run(run, thread)
    messages = client.beta.threads.messages.list(thread_id=thread.id, order="asc")
    latest_message = messages.data[-1].content[0].text.value
    return latest_message


def respond(text: str, chat_history) -> str:
    message = prompt(text)
    chat_history.append((text, message))
    return "", chat_history


def prompt_image(api_key: str, image: np.ndarray, prompt: str) -> str:
    API_URL = "https://api.openai.com/v1/chat/completions"
    headers = compose_headers(api_key=api_key)
    payload = compose_payload(image=image, prompt=prompt)
    response = requests.post(url=API_URL, headers=headers, json=payload).json()

    if "error" in response:
        raise ValueError(response["error"]["message"])
    return response["choices"][0]["message"]["content"]


def cache_image(image: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False) as temp_file:
        image_file_name = temp_file.name
        cv2.imwrite(image_file_name, image)
    return image_file_name


def respond_with_vision(image: np.ndarray, prompt: str, chat_history):
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
        raise gr.Error("An error occurred while generating speech. Please check your API key and come back try again.")

    return response.data[0].url


def chatbot_to_speech(
    chatbot: gr.Chatbot,
    model: Literal["tts-1", "tts-1-hd"],
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    output_file_format: Literal["mp3", "opus", "aac", "flac", ""] = "",
    speed: float = 1.0,
):
    text = chatbot[-1][-1]
    audio = text_to_speech(text, model, voice, output_file_format, speed)
    return audio


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
        raise gr.Error("An error occurred while generating speech. Please check your API key and come back try again.")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_file.write(response.content)

    temp_file_path = temp_file.name

    return temp_file_path


def transcript(audio, model, response_type):
    try:
        client = OpenAI(api_key=openai_key)
        # print(audio)
        audio_file = open(audio, "rb")
        transcriptions = client.audio.transcriptions.create(model=model, file=audio_file, response_format=response_type)
    except Exception as error:
        print(str(error))
        raise gr.Error("An error occurred while generating speech. Please check your API key and come back try again.")

    return transcriptions


##############################################################################################################
# UI
##############################################################################################################


AVATARS = (
    "https://media.roboflow.com/spaces/roboflow_raccoon_full.png",
    "https://media.roboflow.com/spaces/openai-white-logomark.png",
)


# store ui elements in a class
class ui:
    pass


def chat_tab():
    gr.Markdown("# <center> Chat </center>")
    ui.chat_chatbot = gr.Chatbot(height=500, bubble_full_width=False, avatar_images=AVATARS)
    ui.chat_textbox = gr.Textbox(label="Chatbox", placeholder="Type your message here")
    with gr.Row():
        ui.chat_audio_input = gr.Audio(label="Audio Input", source="microphone", type="filepath")
        ui.chat_audio_output = gr.Audio(label="Audio Output", autoplay=True)
    ui.chat_clear_button = gr.ClearButton([ui.chat_textbox, ui.chat_chatbot])


def chat_tab_callbacks():
    ui.chat_textbox.submit(
        fn=respond,
        inputs=[ui.chat_textbox, ui.chat_chatbot],
        outputs=[ui.chat_textbox, ui.chat_chatbot],
    )

    ui.chat_audio_input.stop_recording(
        fn=transcript,
        inputs=[ui.chat_audio_input, ui.stt_model, ui.stt_response_type],
        outputs=[ui.chat_textbox],
    ).then(
        fn=respond,
        inputs=[ui.chat_textbox, ui.chat_chatbot],
        outputs=[ui.chat_textbox, ui.chat_chatbot],
    ).then(
        chatbot_to_speech,
        inputs=[ui.chat_chatbot, ui.tts_model, ui.tts_voice, ui.tts_output_file_format, ui.tts_speed],
        outputs=[ui.chat_audio_output],
    )


def chat_with_vision_tab():
    gr.Markdown("# <center> Chat with Vision </center>")
    with gr.Row():
        ui.vision_webcam = gr.Image(source="webcam", streaming=True)  # Fix uploading flicker
        with gr.Column():
            ui.vision_chatbot = gr.Chatbot(height=500, bubble_full_width=False, avatar_images=AVATARS)
            ui.vision_textbox = gr.Textbox(label="Chatbox", placeholder="Type your message here")
            with gr.Row():
                ui.vision_audio_input = gr.Audio(label="Audio Input", source="microphone", type="filepath")
                ui.vision_audio_output = gr.Audio(label="Audio Output", autoplay=True)
            ui.vision_clear_button = gr.ClearButton([ui.vision_textbox, ui.vision_chatbot])


def chat_with_vision_tab_callbacks():
    ui.vision_textbox.submit(
        fn=respond_with_vision,
        inputs=[ui.vision_webcam, ui.vision_textbox, ui.vision_chatbot],
        outputs=[ui.vision_textbox, ui.vision_chatbot],
    )

    ui.vision_audio_input.stop_recording(
        fn=transcript,
        inputs=[ui.vision_audio_input, ui.stt_model, ui.stt_response_type],
        outputs=[ui.vision_textbox],
    ).then(
        fn=respond_with_vision,
        inputs=[ui.vision_webcam, ui.vision_textbox, ui.vision_chatbot],
        outputs=[ui.vision_textbox, ui.vision_chatbot],
    ).then(
        fn=chatbot_to_speech,
        inputs=[ui.vision_chatbot, ui.tts_model, ui.tts_voice, ui.tts_output_file_format, ui.tts_speed],
        outputs=[ui.vision_audio_output],
    )


def text_to_speech_tab():
    gr.Markdown("# <center> Text to Speech </center>")
    with gr.Row(variant="panel"):
        ui.tts_model = gr.Dropdown(choices=["tts-1", "tts-1-hd"], label="Model", value="tts-1-hd")
        ui.tts_voice = gr.Dropdown(choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"], label="Voice Options", value="alloy")
        ui.tts_output_file_format = gr.Dropdown(choices=["mp3", "opus", "aac", "flac"], label="Output Options", value="mp3")
        ui.tts_speed = gr.Slider(minimum=0.25, maximum=4.0, value=1.0, step=0.01, label="Speed")
    ui.tts_textbox = gr.Textbox(label="Input text", placeholder='Enter your text and then click on the "Text to Speech" button, or press the Enter key.')
    ui.tts_button = gr.Button("Text to Speech")
    ui.tts_output_audio = gr.Audio(label="Speech Output", autoplay=True)


def text_to_speech_tab_callbacks():
    ui.tts_button.click(
        fn=text_to_speech,
        inputs=[ui.tts_textbox, ui.tts_model, ui.tts_voice, ui.tts_output_file_format, ui.tts_speed],
        outputs=ui.tts_output_audio,
    )

    ui.tts_textbox.submit(
        fn=text_to_speech,
        inputs=[ui.tts_textbox, ui.tts_model, ui.tts_voice, ui.tts_output_file_format, ui.tts_speed],
        outputs=ui.tts_output_audio,
    )


def speech_to_text_tab():
    gr.Markdown("# <center> Speech to Text </center>")
    with gr.Row(variant="panel"):
        ui.stt_model = gr.Dropdown(choices=["whisper-1"], label="Model", value="whisper-1")
        ui.stt_response_type = gr.Dropdown(choices=["json", "text", "srt", "verbose_json", "vtt"], label="Response Type", value="text")
    with gr.Row():
        ui.stt_audio_input = gr.Audio(source="microphone", type="filepath")
        ui.stt_file = gr.UploadButton(file_types=[".mp3", ".wav"], label="Select File", type="filepath")
    ui.stt_output_text = gr.Text(label="Output Text")


def speech_to_text_tab_callbacks():
    ui.stt_audio_input.stop_recording(
        fn=transcript,
        inputs=[ui.stt_audio_input, ui.stt_model, ui.stt_response_type],
        outputs=ui.stt_output_text,
    )

    ui.stt_file.upload(fn=transcript, inputs=[ui.stt_file, ui.stt_model, ui.stt_response_type], outputs=ui.stt_output_text)


def image_generation_tab():
    gr.Markdown("# <center> Image Generation </center>")
    with gr.Row(variant="panel"):
        ui.image_model = gr.Dropdown(choices=["dall-e-2", "dall-e-3"], label="Model", value="dall-e-3")
        ui.image_quality = gr.Dropdown(choices=["standard", "hd"], label="Quality", value="standard")
        ui.image_size = gr.Dropdown(choices=["1024x1024", "1792x1024", "1024x1792"], label="Size", value="1024x1024")

    ui.image_textbox = gr.Textbox(label="Input Text", placeholder='Enter your text and then click on the "Image Generate" button, or press the Enter key.')
    ui.image_button = gr.Button("Image Generate")
    ui.image_output_image = gr.Image(label="Image Output")


def image_generation_tab_callbacks():
    ui.image_textbox.submit(
        fn=generate_image,
        inputs=[ui.image_textbox, ui.image_model, ui.image_quality, ui.image_size],
        outputs=ui.image_output_image,
    )

    ui.image_button.click(
        fn=generate_image,
        inputs=[ui.image_textbox, ui.image_model, ui.image_quality, ui.image_size],
        outputs=ui.image_output_image,
    )


def create_ui():
    with gr.Blocks() as gradio:
        with gr.Tab(label="Chat"):
            chat_tab()
        with gr.Tab(label="Chat with Vision"):
            chat_with_vision_tab()
        with gr.Tab(label="Text to Speech"):
            text_to_speech_tab()
        with gr.Tab(label="Speech to Text"):
            speech_to_text_tab()
        with gr.Tab(label="Image Generation"):
            image_generation_tab()

        chat_tab_callbacks()
        chat_with_vision_tab_callbacks()
        text_to_speech_tab_callbacks()
        speech_to_text_tab_callbacks()
        image_generation_tab_callbacks()

    gradio.launch()


if __name__ == "__main__":
    openai_key = get_openai_key()
    client = OpenAI(api_key=openai_key)
    thread = client.beta.threads.create()
    create_ui()
