import os
from queue import Queue
import time
import uuid
import shutil
import subprocess
from typing import Iterator
import cv2

from openai_wrapper import OpenAIClient


IMAGE_CACHE_DIRECTORY = "data/images"
AUDIO_CACHE_DIRECTORY = "data/audio"


def cache_image(data):
    file_name = f"{uuid.uuid4()}.jpeg"
    os.makedirs(IMAGE_CACHE_DIRECTORY, exist_ok=True)
    path = os.path.join(IMAGE_CACHE_DIRECTORY, file_name)
    cv2.imwrite(path, data)

    return path


def cache_audio(data):
    file_name = f"{uuid.uuid4()}.mp3"
    os.makedirs(AUDIO_CACHE_DIRECTORY, exist_ok=True)
    path = os.path.join(AUDIO_CACHE_DIRECTORY, file_name)

    with open(path, "wb") as f:
        f.write(data)

    return path


def is_installed(lib_name: str) -> bool:
    lib = shutil.which(lib_name)
    if lib is None:
        return False
    return True


# def play_audio(audio: bytes, use_ffmpeg: bool = True) -> None:
#     if use_ffmpeg:
#         if not is_installed("ffplay"):
#             message = (
#                 "ffplay from ffmpeg not found, necessary to play audio. "
#                 "On mac you can install it with 'brew install ffmpeg'. "
#                 "On linux and windows you can install it from https://ffmpeg.org/"
#             )
#             raise ValueError(message)
#         args = ["ffplay", "-autoexit", "-", "-nodisp"]
#         proc = subprocess.Popen(
#             args=args,
#             stdout=subprocess.PIPE,
#             stdin=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#         )
#         out, err = proc.communicate(input=audio)
#         proc.poll()
#     else:
#         try:
#             import io

#             import sounddevice as sd
#             import soundfile as sf
#         except ModuleNotFoundError:
#             message = "`pip install sounddevice soundfile` required when `use_ffmpeg=False` "
#             raise ValueError(message)
#         sd.play(*sf.read(io.BytesIO(audio)))
#         sd.wait()


def stream_audio(audio_stream: Iterator[bytes]) -> bytes:
    if not is_installed("mpv"):
        message = (
            "mpv not found, necessary to stream audio. "
            "On mac you can install it with 'brew install mpv'. "
            "On linux and windows you can install it from https://mpv.io/"
        )
        raise ValueError(message)

    mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
    mpv_process = subprocess.Popen(
        mpv_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    audio = b""

    for chunk in audio_stream:
        if chunk is not None:
            mpv_process.stdin.write(chunk)
            mpv_process.stdin.flush()
            audio += chunk

    if mpv_process.stdin:
        mpv_process.stdin.close()
    mpv_process.wait()

    return audio


class AudioQueue:
    def __init__(self, is_streaming: bool = False):
        self.is_streaming = is_streaming
        self.audio = Queue()
        self.text = Queue()


def generate_sentences(queue: AudioQueue, client: OpenAIClient):
    sentence = ""
    while queue.is_streaming:
        time.sleep(0.05)
        while not queue.text.empty():
            chunk = queue.text.get()
            sentence += chunk
            if chunk and chunk[-1] in ".!?":  # TODO: add a better way to detect end of sentence
                audio = client.text_to_speech_streaming(sentence)
                queue.audio.put(audio)
                sentence = ""


def stream_audio_generator(queue: AudioQueue) -> Iterator[bytes]:
    while queue.is_streaming:
        time.sleep(0.05)
        while not queue.audio.empty():
            sentence_audio = queue.audio.get()
            yield sentence_audio


def stream_sentences(queue: Queue):
    for audio in stream_audio_generator(queue):
        stream_audio(audio.iter_bytes())
