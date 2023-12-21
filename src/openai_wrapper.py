from typing import Iterator
from typing_extensions import Union, Literal
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk

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


class OpenAIClient:
    def __init__(self, client: OpenAI):
        self.client = client
        self.history = Messages()

    def chat(
        self,
        prompt: str,
        model: Literal["gpt-4", "gpt-4-1106-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"] = "gpt-3.5-turbo",
    ) -> str:
        self.history.add_user_message(prompt)
        completion = self.client.chat.completions.create(model=model, messages=self.history.messages)
        response = completion.choices[0].message.content
        self.history.add_assistant_message(response)
        return response

    def chat_streaming(
        self, prompt: str, model: Literal["gpt-4", "gpt-4-1106-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"] = "gpt-3.5-turbo"
    ) -> Stream[ChatCompletionChunk]:
        self.history.add_user_message(prompt)
        stream = self.client.chat.completions.create(stream=True, model=model, messages=self.history.messages)
        return stream

    def chat_stream_generator(self, stream: Stream[ChatCompletionChunk]) -> Iterator[str]:
        full_text = ""
        for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            full_text += text
            yield text
        self.history.add_assistant_message(full_text)

    def chat_stream_printer(self, stream: Stream[ChatCompletionChunk]):
        for part in stream:
            chunk = part.choices[0].delta.content
            print(chunk or "", end="", flush=True)

    def speech_to_text(
        self,
        audio: str,
        model: Literal["whisper-1"] = "whisper-1",
        response_type: Literal["json", "text", "srt", "verbose_json", "vtt"] = "text",
    ) -> str:
        try:
            audio_file = open(audio, "rb")
            transcription = self.client.audio.transcriptions.create(model=model, file=audio_file, response_format=response_type)
            return transcription
        except Exception as error:
            print(f"Speech to text error: {error}")
            return ""

    def text_to_speech(
        self,
        text: str,
        model: Literal["tts-1", "tts-1-hd"] = "tts-1-hd",
        voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "alloy",
        output_file_format: Literal["mp3", "opus", "aac", "flac", ""] = "mp3",
        speed: float = 1.0,
    ) -> str:
        try:
            response = self.client.audio.speech.create(input=text, model=model, voice=voice, response_format=output_file_format, speed=speed)
            # Assuming 'cache_audio' is a function defined elsewhere that saves the audio content and returns the file path.
            audio_path = cache_audio(response.content)
            return audio_path
        except Exception as error:
            print(f"Text to speech error: {error}")
            return ""

    def generate_image(
        self,
        prompt: str,
        model: Literal["dall-e-2", "dall-e-3"] = "dall-e-3",
        quality: Literal["standard", "hd"] = "hd",
        size: Literal["1024x1024", "1024x1792", "1792x1024"] = "1024x1024",
        style: Literal["vivid", "natural"] = "vivid",
    ) -> str:
        try:
            response = self.client.images.generate(prompt=prompt, model=model, quality=quality, size=size, style=style, n=1)
            url = response.data[0].url
            return url
        except Exception as error:
            print(f"Dall-e error: {error}")
            return ""


# ai = OpenAIClient(OpenAI(api_key=os.environ.get("OPENAI_API_KEY")))
# stream = ai.chat_streaming("what can you do>")

# printer = ai.chat_stream_generator(stream)
# for chunk in printer:
#     print(chunk)
