from typing_extensions import Union, Literal
from openai import OpenAI

from utils import *


class Messages:
    def __init__(self, system_prompt: str = None):
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def add_system_message(self, content):
        self.messages.append({"role": "system", "content": content})

    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content):
        self.messages.append({"role": "assistant", "content": content})

    def remove_last_request(self):
        self.messages.pop()
        self.messages.pop()

    def clear_messages(self):
        self.messages = []

    def get_last_message(self):
        return self.messages[-1]["content"]

    def get_messages(self):
        return self.messages


def chat(
    client: OpenAI,
    prompt: Union[str, list[dict[str, str]], Messages],
    model: Literal["gpt-4", "gpt-4-1106-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"] = "gpt-3.5-turbo",
) -> str:
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, Messages):
        messages = prompt.add_user_message(prompt)
    else:
        messages = prompt
    print(messages)
    completion = client.chat.completions.create(model=model, messages=messages)
    response = completion.choices[0].message.content
    if isinstance(prompt, Messages):
        prompt.add_assistant_message(response)
    return response


def chat_stream(client: OpenAI, prompt: str, model: Literal["gpt-4", "gpt-4-1106-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"] = "gpt-3.5-turbo"):
    stream = client.chat.completions.create(stream=True, model=model, messages=[{"role": "user", "content": prompt}])
    return stream


def print_chat_stream(stream):
    for part in stream:
        chunk = part.choices[0].delta.content
        print(chunk or "", end="", flush=True)


def speech_to_text(
    client: OpenAI,
    audio: str,
    model: Literal["whisper-1"] = "whisper-1",
    response_type: Literal["json", "text", "srt", "verbose_json", "vtt"] = "text",
) -> str:
    try:
        audio_file = open(audio, "rb")
        transcription = client.audio.transcriptions.create(model=model, file=audio_file, response_format=response_type)
        return transcription
    except Exception as error:
        print(f"Speech to text error: {error}")
        return ""


def text_to_speech(
    client: OpenAI,
    text: str,
    model: Literal["tts-1", "tts-1-hd"] = "tts-1-hd",
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "alloy",
    output_file_format: Literal["mp3", "opus", "aac", "flac", ""] = "mp3",
    speed: float = 1.0,
) -> str:
    try:
        response = client.audio.speech.create(input=text, model=model, voice=voice, response_format=output_file_format, speed=speed)
        # Assuming 'cache_audio' is a function defined elsewhere that saves the audio content and returns the file path.
        audio_path = cache_audio(response.content)
        return audio_path
    except Exception as error:
        print(f"Text to speech error: {error}")
        return ""


def generate_image(
    client: OpenAI,
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
        print(f"Dall-e error: {error}")
        return ""
