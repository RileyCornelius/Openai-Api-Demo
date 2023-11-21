from openai import OpenAI
import time


# Pretty printing helper
def pretty_print(messages):
    print("# Messages")
    for m in messages:
        print(f"{m.role}: {m.content[0].text.value}")
    print()


# Waiting in a loop
def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


MATH_ASSISTANT_ID = "asst_IO7ySDCqFNGz7yuZrov8WLsX"

client = OpenAI()


def submit_message(assistant_id, thread, user_message):
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=user_message)
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )


def get_response(thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")


def create_thread_and_run(user_input):
    thread = client.beta.threads.create()
    run = submit_message(MATH_ASSISTANT_ID, thread, user_input)
    return thread, run


def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


def pretty_print(messages):
    print("# Messages")
    for m in messages:
        print(f"{m.role}: {m.content[0].text.value}")
    print()


def prompt(client: OpenAI, thread, prompt: str) -> str:
    BAXTER_ASSISTANT_ID = "asst_IO7ySDCqFNGz7yuZrov8WLsX"
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=prompt)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=BAXTER_ASSISTANT_ID)
    wait_on_run(run, thread)
    messages = client.beta.threads.messages.list(thread_id=thread.id, order="asc")
    latest_message = messages.data[-1]
    print(latest_message.content[0].text.value)
    print("---")
    return latest_message.content[0].text.value


thread = client.beta.threads.create()
prompt(client, thread, "what is TACAM")
prompt(client, thread, "what is the Smart Factory")


# thread1, run1 = create_thread_and_run("what is TACAM")
# run1 = wait_on_run(run1, thread1)
# pretty_print(get_response(thread1))
