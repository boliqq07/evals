from typing import Any, Optional, Union
import os
import requests

from openai import OpenAI

from evals.api import CompletionFn, CompletionResult
from evals.base import CompletionFnSpec
from evals.prompt.base import (
    ChatCompletionPrompt,
    CompletionPrompt,
    OpenAICreateChatPrompt,
    OpenAICreatePrompt,
    Prompt,
)
from evals.record import record_sampling
from evals.utils.api_utils import (
    request_with_timeout
)

default_prompts = {
    "activity": "请汇总文献中全部抑制剂(分子请分别用名字和SMILES表达)的结合活性、活性种类（IC50, EC50, TC50, Ki, Kd中的一个），并备注每类结合活性的实验手段。以json格式输出，活性和活性类型的字段名分别为 \"Affinity\" 和 \"Affinity_type\"",
}


class Struct:
    def __init__(self, **entries):
        self.__dict__.update({k: self._wrap(v) for k, v in entries.items()})

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(**value) if isinstance(value, dict) else value

    def __repr__(self):
        return '<%s>' % str('\n '.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))


class BaseCompletionResult(CompletionResult):
    def __init__(self, raw_data: Any, prompt: Any):
        self.raw_data = Struct(raw_data) if type(raw_data) == dict else raw_data
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        raise NotImplementedError


class ZhishuCompletionResult(BaseCompletionResult):
    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data:
            for choice in self.raw_data.choices:
                if choice.message.content is not None:
                    completions.append(choice.text)
        return completions


class ZhishuCompletionFn(CompletionFn):
    def __init__(
            self,
            model: Optional[str] = None,
            api_base: Optional[str] = None,
            api_key: Optional[str] = None,
            n_ctx: Optional[int] = None,
            extra_options: Optional[dict] = {},
            **kwargs,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.n_ctx = n_ctx
        self.extra_options = extra_options

    def __call__(
            self,
            prompt: Union[str, OpenAICreateChatPrompt],
            **kwargs,
    ) -> ZhishuCompletionResult:
        if not isinstance(prompt, Prompt):
            assert (
                    isinstance(prompt, str)
                    or (isinstance(prompt, list) and all(isinstance(token, int) for token in prompt))
                    or (isinstance(prompt, list) and all(isinstance(token, str) for token in prompt))
                    or (isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt))
            ), f"Got type {type(prompt)}, with val {type(prompt[0])} for prompt, expected str or list[int] or list[str] or list[dict[str, str]]"

            prompt = CompletionPrompt(
                raw_prompt=prompt,
            )

        openai_create_prompt: OpenAICreatePrompt = prompt.to_formatted_prompt()

        url = f"https://api.zhishuyun.com/openai/gpt-4-all?token={self.api_key or os.environ['ZHISHU_API_KEY']}"
        headers = {
            "content-type": "application/json"
        }
        payload = {
            "model": "gpt-4-all",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{kwargs['file_link']} {prompt}"}
            ]
        }

        result = request_with_timeout(requests.post, url, json=payload, headers=headers)

        result = ZhishuCompletionResult(raw_data=result.json(), prompt=openai_create_prompt)
        record_sampling(prompt=result.prompt, sampled=result.get_completions())
        return result
