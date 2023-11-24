from ui import create_ui
from openai_client import set_openai_key


if __name__ == "__main__":
    set_openai_key()
    create_ui()
