import os

from typing_extensions import Literal
from dotenv import load_dotenv
from openai import OpenAI

from utils import *


openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)


def set_openai_key():
    if load_dotenv():
        openai_key = os.getenv("OPENAI_API_KEY")
        client.api_key = openai_key


def chat(
    prompt,
    model: Literal[
        "gpt-4",
        "gpt-4-1106-preview",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
    ] = "gpt-3.5-turbo",
) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    response = completion.choices[0].message.content
    return response


def chat_stream(
    prompt,
    model: Literal[
        "gpt-4",
        "gpt-4-1106-preview",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
    ] = "gpt-3.5-turbo",
):
    stream = client.chat.completions.create(
        stream=True,
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return stream


def print_chat_stream(stream):
    for part in stream:
        chunk = part.choices[0].delta.content
        print(chunk or "", end="", flush=True)


def generate_image(
    prompt: str,
    model: Literal["dall-e-2", "dall-e-3"],
    quality: Literal["standard", "hd"],
    size: Literal["1024x1024", "1024x1792", "1792x1024"],
) -> str:
    try:
        response = client.images.generate(
            prompt=prompt,
            model=model,
            quality=quality,
            size=size,
            n=1,
        )
    except Exception as error:
        print("Dall-e error: " + error.code)
        return ""

    url = response.data[0].url
    return url


def speech_to_text(
    audio,
    model: Literal["whisper-1"] = "whisper-1",
    response_type: Literal["json", "text", "srt", "verbose_json", "vtt"] = "text",
) -> str:
    try:
        audio_file = open(audio, "rb")
        transcriptions = client.audio.transcriptions.create(model=model, file=audio_file, response_format=response_type)
    except Exception as error:
        print(print("Speech to text error: " + error.code))
        return ""

    return transcriptions


def text_to_speech(
    text: str,
    model: Literal["tts-1", "tts-1-hd"],
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "alloy",
    output_file_format: Literal["mp3", "opus", "aac", "flac", ""] = "mp3",
    speed: float = 1.0,
) -> str:
    try:
        response = client.audio.speech.create(
            input=text,
            model=model,
            voice=voice,
            response_format=output_file_format,
            speed=speed,
        )
    except Exception as error:
        print("Text to speech error: " + error.code)
        return ""

    audio_path = cache_audio(response.content)
    return audio_path


def chatbot_to_speech(
    chatbot,
    model: Literal["tts-1", "tts-1-hd"],
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    output_file_format: Literal["mp3", "opus", "aac", "flac", ""],
    speed: float,
):
    text = chatbot[-1][-1]
    audio = text_to_speech(text, model, voice, output_file_format, speed)
    return audio
