import os

from typing_extensions import Literal
from dotenv import load_dotenv
from openai import OpenAI

from utils import *

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)


def chat(prompt, model: Literal["gpt-4", "gpt-4-1106-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"] = "gpt-3.5-turbo") -> str:
    completion = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    response = completion.choices[0].message.content
    return response


def chat_stream(prompt, model: Literal["gpt-4", "gpt-4-1106-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"] = "gpt-3.5-turbo"):
    stream = client.chat.completions.create(stream=True, model=model, messages=[{"role": "user", "content": prompt}])
    return stream


def print_chat_stream(stream):
    for part in stream:
        chunk = part.choices[0].delta.content
        print(chunk or "", end="", flush=True)


def speech_to_text(
    audio,
    model: Literal["whisper-1"] = "whisper-1",
    response_type: Literal["json", "text", "srt", "verbose_json", "vtt"] = "text",
) -> str:
    try:
        audio_file = open(audio, "rb")
        transcription = client.audio.transcriptions.create(model=model, file=audio_file, response_format=response_type)
        return transcription
    except Exception as error:
        print(f"Speech to text error: {error.code}")
        return ""


def text_to_speech(
    text: str,
    model: Literal["tts-1", "tts-1-hd"] = "tts-1-hd",
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "alloy",
    output_file_format: Literal["mp3", "opus", "aac", "flac", ""] = "mp3",
    speed: float = 1.0,
) -> str:
    try:
        response = client.audio.speech.create(input=text, model=model, voice=voice, response_format=output_file_format, speed=speed)
        audio_path = cache_audio(response.content)
        return audio_path
    except Exception as error:
        print(f"Text to speech error: {error.code}")
        return ""


def generate_image(
    prompt: str,
    model: Literal["dall-e-2", "dall-e-3"] = "dall-e-3",
    quality: Literal["standard", "hd"] = "hd",
    size: Literal["1024x1024", "1024x1792", "1792x1024"] = "1024x1024",
    style: Literal["vivid", "natural"] = "vivid",
) -> str:
    try:
        response = client.images.generate(prompt=prompt, model=model, quality=quality, size=size, style=style, n=1)
        url = response.data[0].url
        return url
    except Exception as error:
        print(f"Dall-e error:  {error.code}")
        return ""
