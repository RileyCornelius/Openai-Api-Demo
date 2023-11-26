import os
from dotenv import load_dotenv
from openai import OpenAI
from openai_assistant import Assistant
from ui import UI

if __name__ == "__main__":
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    assistant_id = os.getenv("OPENAI_ASSISTANT_ID")

    client = OpenAI(api_key=openai_key)
    assistant = Assistant(openai_key)
    assistant.set_assistant(assistant_id)

    ui = UI(assistant, client)
    ui.launch()
