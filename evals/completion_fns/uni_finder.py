"""
Extending Completion Functions with Embeddings-based retrieval from a fetched dataset
"""
import os
import time
import requests
from typing import Any, Optional, Union

import numpy as np
from openai import OpenAI

import pandas as pd

from evals.api import CompletionFn, CompletionResult
from evals.record import record_sampling
from evals.utils.api_utils import (
    request_with_timeout
)


class UniFinderCompletionResult(CompletionResult):
    def __init__(self, response: str) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]


class UniFinderCompletionFn(CompletionFn):
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
        self.api_base = api_base or os.environ.get("UNIFINDER_API_BASE")
        self.api_key = api_key or os.environ.get("UNIFINDER_API_KEY")
        self.n_ctx = n_ctx
        self.extra_options = extra_options

    def __call__(self, prompt: Union[str, list[dict]], **kwargs: Any) -> UniFinderCompletionResult:
        """
        Args:
            prompt: The prompt to complete, in either text string or Chat format.
            kwargs: Additional arguments to pass to the completion function call method.
        """

        pdf_token = []
        if "file_name" in kwargs:
            url = f"{self.api_base}/api/external/upload_pdf"
            pdf_parse_mode = 'fast'  # or 'precise', 指定使用的pdf解析版本
            files = {'file': open(kwargs["file_name"], 'rb')}
            data = {
                'pdf_parse_mode': pdf_parse_mode,
                'api_key': self.api_key
            }
            response = requests.post(url, data=data, files=files).json()
            pdf_id = response['pdf_token']  # 获得pdf的id，表示上传成功，后续可以使用这个id来指定pdf
            pdf_token.append(pdf_id)

        url = f"{self.api_base}/api/external/chatpdf"

        payload = {
            "model_engine": self.model,
            "pdf_token": pdf_token,
            "query": prompt,
            'api_key': self.api_key
        }
        response = requests.post(url, json=payload).json()
        answer = response['answer']
        print(answer)
        record_sampling(prompt=prompt, sampled=answer)
        return UniFinderCompletionResult(answer)
