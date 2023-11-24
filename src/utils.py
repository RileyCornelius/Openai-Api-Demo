import os
import uuid

import cv2

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
