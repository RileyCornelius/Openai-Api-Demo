import os
import sys
import time
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


def transcript_audio(audio):
    transcriptions = transcript(audio, "whisper-1", "text")
    return transcriptions


def transcript(audio, model, response_type):
    try:
        client = OpenAI(api_key=openai_key)
        # print(audio)
        audio_file = open(audio, "rb")
        transcriptions = client.audio.transcriptions.create(
            model=model, file=audio_file, response_format=response_type
        )
    except Exception as error:
        print(str(error))
        raise gr.Error("An error occurred while generating speech. Please check your API key and come back try again.")

    return transcriptions


##############################################################################################################
# UI
##############################################################################################################


def chat_tab():
    gr.Markdown("# <center> Chat </center>")
    with gr.Column():
        AVATARS = (
            "https://media.roboflow.com/spaces/roboflow_raccoon_full.png",
            "https://media.roboflow.com/spaces/openai-white-logomark.png",
        )
        chatbot = gr.Chatbot(height=500, bubble_full_width=False, avatar_images=AVATARS)
        message_textbox = gr.Textbox(label="Chatbox", placeholder="Type your message here")
        clear_button = gr.ClearButton([message_textbox, chatbot])

    message_textbox.submit(
        fn=respond,
        inputs=[message_textbox, chatbot],
        outputs=[message_textbox, chatbot],
    )


def chat_with_vision_tab():
    gr.Markdown("# <center> Chat with Vision </center>")
    with gr.Column():
        with gr.Row():
            webcam = gr.Image(source="webcam", streaming=True)  # Fix uploading flicker
            with gr.Column():
                AVATARS = (
                    "https://media.roboflow.com/spaces/roboflow_raccoon_full.png",
                    "https://media.roboflow.com/spaces/openai-white-logomark.png",
                )
                chatbot = gr.Chatbot(height=500, bubble_full_width=False, avatar_images=AVATARS)
                message_textbox = gr.Textbox(label="Chatbox", placeholder="Type your message here")
                audio_input = gr.Audio(label="Audio Input", source="microphone", type="filepath")
                audio_output = gr.Audio(label="Audio Output", autoplay=True)
                clear_button = gr.ClearButton([message_textbox, chatbot])

        with gr.Row():
            model = gr.Dropdown(choices=["tts-1", "tts-1-hd"], label="Model", value="tts-1")
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
            speed = gr.Slider(minimum=0.25, maximum=4.0, value=1.0, step=0.01, label="Speed")

    audio_input.stop_recording(
        fn=transcript_audio,
        inputs=[audio_input],
        outputs=[message_textbox],
    ).then(
        fn=respond_with_vision,
        inputs=[webcam, message_textbox, chatbot],
        outputs=[message_textbox, chatbot],
    ).then(chatbot_to_speech, inputs=[chatbot, model, voice, output_file_format, speed], outputs=[audio_output])

    message_textbox.submit(
        fn=respond_with_vision,
        inputs=[webcam, message_textbox, chatbot],
        outputs=[message_textbox, chatbot],
    ).then(chatbot_to_speech, inputs=[chatbot, model, voice, output_file_format, speed], outputs=[audio_output])


def text_to_speech_tab():
    gr.Markdown("# <center> Text to Speech </center>")
    with gr.Row(variant="panel"):
        model = gr.Dropdown(choices=["tts-1", "tts-1-hd"], label="Model", value="tts-1")
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
        speed = gr.Slider(minimum=0.25, maximum=4.0, value=1.0, step=0.01, label="Speed")

    text = gr.Textbox(
        label="Input text",
        placeholder='Enter your text and then click on the "Text-To-Speech" button, ' "or simply press the Enter key.",
    )
    btn = gr.Button("Text-To-Speech")
    output_audio = gr.Audio(label="Speech Output", autoplay=True)

    btn.click(
        fn=text_to_speech,
        inputs=[text, model, voice, output_file_format, speed],
        outputs=output_audio,
    )
    text.submit(
        fn=text_to_speech,
        inputs=[text, model, voice, output_file_format, speed],
        outputs=output_audio,
    )


def speech_to_text_tab():
    gr.Markdown("# <center> Speech to Text </center>")
    with gr.Row(variant="panel"):
        model = gr.Dropdown(choices=["whisper-1"], label="Model", value="whisper-1")
        response_type = gr.Dropdown(
            choices=["json", "text", "srt", "verbose_json", "vtt"],
            label="Response Type",
            value="text",
        )

    with gr.Row():
        audio_input = gr.Audio(source="microphone", type="filepath")
        file = gr.UploadButton(
            file_types=[".mp3", ".wav"],
            label="Select File",
            type="filepath",
        )

    output_text = gr.Text(label="Output Text")
    audio_input.stop_recording(
        fn=transcript,
        inputs=[audio_input, model, response_type],
        outputs=output_text,
    )
    file.upload(fn=transcript, inputs=[file, model, response_type], outputs=output_text)


def image_generation_tab():
    gr.Markdown("# <center> Image Generation </center>")
    with gr.Row(variant="panel"):
        model = gr.Dropdown(choices=["dall-e-2", "dall-e-3"], label="Model", value="dall-e-3")
        quality = gr.Dropdown(choices=["standard", "hd"], label="Quality", value="standard")
        size = gr.Dropdown(
            choices=["1024x1024", "1792x1024", "1024x1792"],
            label="Size",
            value="1024x1024",
        )

    text = gr.Textbox(
        label="Input Text",
        placeholder='Enter your text and then click on the "Image Generate" button, ' "or simply press the Enter key.",
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


def ui():
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

    gradio.launch()


if __name__ == "__main__":
    openai_key = get_openai_key()  # global api key
    client = OpenAI(api_key=openai_key)
    thread = client.beta.threads.create()
    ui()
