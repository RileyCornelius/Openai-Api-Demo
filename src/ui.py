import gradio as gr
from typing_extensions import Literal

from openai_client import *
from openai_vision import *
from openai_assistant import *

AVATARS = (
    "https://media.roboflow.com/spaces/roboflow_raccoon_full.png",
    "https://media.roboflow.com/spaces/openai-white-logomark.png",
)


assistant_id = os.getenv("OPENAI_ASSISTANT_ID")
assistant = Assistant(openai_key)
assistant.set_assistant(assistant_id)


def chatbot_response(prompt, chat_history, model):
    response = chat(prompt, model)
    chat_history.append((prompt, response))
    return "", chat_history


def chatbot_text_to_speech(chat_history, model, voice, output_file_format, speed):
    text = chat_history[-1][-1]
    audio = text_to_speech(text, model, voice, output_file_format, speed)
    return audio


def chatbot_assistant_respond(prompt, chat_history):
    assistant.create_message(prompt)
    assistant.create_run()
    if assistant.wait_for_run():
        message = assistant.get_last_message()
        chat_history.append((prompt, message))
    return "", chat_history


##########################################################################################
# UI
##########################################################################################


class ui:  # stores ui elements to be referenced in callbacks
    pass


def chat_tab():
    gr.Markdown("# <center> Chat </center>")
    ui.chat_chatbot = gr.Chatbot(height=500, bubble_full_width=False, avatar_images=AVATARS)
    with gr.Row():
        ui.chat_textbox = gr.Textbox(label="Chatbox", placeholder="Type your message here", scale=3)
        ui.chat_model = gr.Dropdown(choices=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-1106-preview"], label="Model", value="gpt-3.5-turbo")
    with gr.Row():
        ui.chat_audio_input = gr.Audio(label="Audio Input", source="microphone", type="filepath")
        ui.chat_audio_output = gr.Audio(label="Audio Output", autoplay=True)
    ui.chat_clear_button = gr.ClearButton([ui.chat_textbox, ui.chat_chatbot])


def chat_tab_callbacks():
    ui.chat_textbox.submit(
        fn=chatbot_response,
        inputs=[ui.chat_textbox, ui.chat_chatbot, ui.chat_model],
        outputs=[ui.chat_textbox, ui.chat_chatbot],
    )

    ui.chat_audio_input.stop_recording(
        fn=speech_to_text,
        inputs=[ui.chat_audio_input, ui.stt_model, ui.stt_response_type],
        outputs=[ui.chat_textbox],
    ).then(
        fn=chatbot_response,
        inputs=[ui.chat_textbox, ui.chat_chatbot, ui.chat_model],
        outputs=[ui.chat_textbox, ui.chat_chatbot],
    ).then(
        chatbot_text_to_speech,
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
        fn=speech_to_text,
        inputs=[ui.vision_audio_input, ui.stt_model, ui.stt_response_type],
        outputs=[ui.vision_textbox],
    ).then(
        fn=respond_with_vision,
        inputs=[ui.vision_webcam, ui.vision_textbox, ui.vision_chatbot],
        outputs=[ui.vision_textbox, ui.vision_chatbot],
    ).then(
        fn=chatbot_text_to_speech,
        inputs=[ui.vision_chatbot, ui.tts_model, ui.tts_voice, ui.tts_output_file_format, ui.tts_speed],
        outputs=[ui.vision_audio_output],
    )


def assistant_tab():
    gr.Markdown("# <center> Assistant </center>")
    ui.assistant_chatbot = gr.Chatbot(height=500, bubble_full_width=False, avatar_images=AVATARS)
    ui.assistant_textbox = gr.Textbox(label="chatbox", placeholder="Type your message here")
    with gr.Row():
        ui.assistant_audio_input = gr.Audio(label="Audio Input", source="microphone", type="filepath")
        ui.assistant_audio_output = gr.Audio(label="Audio Output", autoplay=True)
    ui.assistant_clear_button = gr.ClearButton([ui.assistant_textbox, ui.assistant_chatbot])


def assistant_tab_callbacks():
    ui.assistant_textbox.submit(
        fn=chatbot_assistant_respond,
        inputs=[ui.assistant_textbox, ui.assistant_chatbot],
        outputs=[ui.assistant_textbox, ui.assistant_chatbot],
    )

    ui.assistant_audio_input.stop_recording(
        fn=speech_to_text,
        inputs=[ui.assistant_audio_input, ui.stt_model, ui.stt_response_type],
        outputs=[ui.assistant_textbox],
    ).then(
        fn=chatbot_assistant_respond,
        inputs=[ui.assistant_textbox, ui.assistant_chatbot],
        outputs=[ui.assistant_textbox, ui.assistant_chatbot],
    ).then(
        chatbot_text_to_speech,
        inputs=[ui.assistant_chatbot, ui.tts_model, ui.tts_voice, ui.tts_output_file_format, ui.tts_speed],
        outputs=[ui.assistant_audio_output],
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
        fn=speech_to_text,
        inputs=[ui.stt_audio_input, ui.stt_model, ui.stt_response_type],
        outputs=ui.stt_output_text,
    )

    ui.stt_file.upload(fn=speech_to_text, inputs=[ui.stt_file, ui.stt_model, ui.stt_response_type], outputs=ui.stt_output_text)


def image_generation_tab():
    gr.Markdown("# <center> Image Generation </center>")
    with gr.Row(variant="panel"):
        ui.image_model = gr.Dropdown(choices=["dall-e-2", "dall-e-3"], label="Model", value="dall-e-3")
        ui.image_quality = gr.Dropdown(choices=["standard", "hd"], label="Quality", value="standard")
        ui.image_size = gr.Dropdown(choices=["1024x1024", "1792x1024", "1024x1792"], label="Size", value="1024x1024")
        ui.image_style = gr.Dropdown(choices=["vivid", "natural"], label="Style", value="vivid")

    ui.image_textbox = gr.Textbox(label="Input Text", placeholder='Enter your text and then click on the "Image Generate" button, or press the Enter key.')
    ui.image_button = gr.Button("Image Generate")
    ui.image_output_image = gr.Image(label="Image Output")


def image_generation_tab_callbacks():
    ui.image_textbox.submit(
        fn=generate_image,
        inputs=[ui.image_textbox, ui.image_model, ui.image_quality, ui.image_size, ui.image_style],
        outputs=ui.image_output_image,
    )

    ui.image_button.click(
        fn=generate_image,
        inputs=[ui.image_textbox, ui.image_model, ui.image_quality, ui.image_size, ui.image_style],
        outputs=ui.image_output_image,
    )


def create_ui():
    with gr.Blocks() as gradio:
        with gr.Tab(label="Chat"):
            chat_tab()
        with gr.Tab(label="Chat with Vision"):
            chat_with_vision_tab()
        with gr.Tab(label="Assistant"):
            assistant_tab()
        with gr.Tab(label="Text to Speech"):
            text_to_speech_tab()
        with gr.Tab(label="Speech to Text"):
            speech_to_text_tab()
        with gr.Tab(label="Image Generation"):
            image_generation_tab()

        chat_tab_callbacks()
        chat_with_vision_tab_callbacks()
        assistant_tab_callbacks()
        text_to_speech_tab_callbacks()
        speech_to_text_tab_callbacks()
        image_generation_tab_callbacks()

    gradio.launch()


if __name__ == "__main__":
    create_ui()
