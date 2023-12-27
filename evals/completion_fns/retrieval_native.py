"""
Extending Completion Functions with Embeddings-based retrieval from a fetched dataset
"""
import os
from ast import literal_eval
import time
from typing import Any, Optional, Union

import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
import pandas as pd

from evals.api import CompletionFn, CompletionResult
from evals.prompt.base import ChatCompletionPrompt, CompletionPrompt
from evals.record import record_sampling


class RetrievalCompletionResult(CompletionResult):
    def __init__(self, response: str) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]


class OpenAIRetrievalCompletionFn(CompletionFn):
    """
    This Completion Function uses embeddings to retrieve the top k relevant docs from a dataset to the prompt, then adds them to the context before calling the completion.
    """

    def __init__(
            self,
            model: Optional[str] = None,
            instructions: Optional[str] = "You are a helpful assistant on extracting information from files.",
            api_base: Optional[str] = None,
            api_key: Optional[str] = None,
            n_ctx: Optional[int] = None,
            extra_options: Optional[dict] = {},
            **kwargs
    ):
        self.model = model
        self.instructions = instructions
        self.api_base = api_base
        self.api_key = api_key
        self.n_ctx = n_ctx
        self.extra_options = extra_options

    def __call__(self, prompt: Union[str, list[dict]], **kwargs: Any) -> RetrievalCompletionResult:
        """
        Args:
            prompt: The prompt to complete, in either text string or Chat format.
            kwargs: Additional arguments to pass to the completion function call method.
        """

        assert "file_name" in kwargs, "Must provide a file_name to retrieve."

        file = client.files.create(file=open(kwargs["file_name"], "rb"), purpose='assistants')

        #  Create an Assistant (Note model="gpt-3.5-turbo-1106" instead of "gpt-4-1106-preview")
        assistant = client.beta.assistants.create(
            name="File Assistant",
            instructions=self.instructions,
            model=self.model,
            tools=[{"type": "retrieval"}],
            file_ids=[file.id]
        )

        #  Create a Thread
        thread = client.beta.threads.create()

        # Add a Message to a Thread
        print(prompt)
        message = client.beta.threads.messages.create(thread_id=thread.id, role="user",
                                                      content=prompt
                                                      )

        # Run the Assistant
        run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)
        print(run.model_dump_json(indent=4))

        # If run is 'completed', get messages and print
        while True:
            # Retrieve the run status
            run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            time.sleep(10)
            if run_status.status == 'completed':
                messages = client.beta.threads.messages.list(thread_id=thread.id)
                answer = messages.data[0].content[0].text.value
                break
            else:
                ### sleep again
                time.sleep(2)
        print(answer)
        record_sampling(prompt=prompt, sampled=answer)
        return RetrievalCompletionResult(answer)
