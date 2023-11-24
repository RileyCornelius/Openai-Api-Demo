import time
from openai_client import client

thread = client.beta.threads.create()


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
