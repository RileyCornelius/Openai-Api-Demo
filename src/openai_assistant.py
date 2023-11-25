import time

from openai import OpenAI


class Assistant:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key).beta
        self.thread_id = None
        self.assistant_id = None
        self.run_id = None
        self.create_thread()

    def create_thread(self):
        thread = self.client.threads.create()
        self.thread_id = thread.id

    def create_assistant(self, instructions, name=None, model="gpt-4-1106-preview"):
        assistant = self.client.assistants.create(instructions=instructions, name=name, model=model)
        self.assistant_id = assistant.id

    def set_assistant(self, assistant_id):
        self.assistant_id = assistant_id

    def create_message(self, content):
        self.client.threads.messages.create(
            content=content,
            thread_id=self.thread_id,
            role="user",
        )

    def create_run(self, model=None, instructions=None):
        run = self.client.threads.runs.create(thread_id=self.thread_id, assistant_id=self.assistant_id, model=model, instructions=instructions)
        self.run_id = run.id

    def wait_for_run(self):
        while True:
            run = self.client.threads.runs.retrieve(thread_id=self.thread_id, run_id=self.run_id)
            if run.status in {"queued", "in_progress"}:
                time.sleep(0.5)
            elif run.status in {"failed", "expired"}:
                return False
            elif run.status == "completed":
                return True
            else:
                raise Exception(f"Unexpected status: {run.status}")

    def get_messages(self):
        messages = self.client.threads.messages.list(thread_id=self.thread_id, order="asc")
        return messages

    def get_last_message(self):
        messages = self.get_messages()
        latest_message = messages.data[-1].content[0].text.value
        return latest_message
