from io import StringIO
import json
import os
import re

from typing import List, Optional

import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider

from urllib.parse import parse_qs, urlparse

from datasets import load_dataset
import pandas as pd
from pydantic import BaseModel

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.formatting import make_abc
from evals.record import RecorderBase, record_match

code_pattern = r"```[\s\S]*?\n([\s\S]+)\n```"
json_pattern = r"```json[\s\S]*?\n([\s\S]+)\n```"
csv_pattern = r"```csv[\s\S]*?\n([\s\S]+)\n```"


def parse_table_multiindex(table: pd.DataFrame) -> pd.DataFrame:
    """
    Parse a table with multiindex columns.
    """

    df = table.copy()
    coltypes = {col: type(df[col].iloc[0]) for col in df.columns}
    for col, ctype in coltypes.items():
        if ctype == str:
            if ":" in df[col].iloc[0] and "," in df[col].iloc[0]:
                df[col] = [{key: value for key, value in [pair.split(": ") for pair in data.split(", ")]} for data in
                           df[col]]
                coltypes[col] = dict
    dfs = []

    for col, ctype in coltypes.items():
        if ctype == dict:
            d = pd.DataFrame(df.pop(col).tolist())
            dfs.append(d)
    df = pd.concat([df] + dfs, axis=1)
    return df


def init_oss():
    """
    Initialize OSS client.
    """
    # Please set OSS_ACCESS_KEY_ID & OSS_ACCESS_KEY_SECRET in your environment variables.
    auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())

    # 设置 Endpoint
    endpoint = 'https://oss-cn-beijing.aliyuncs.com'

    # 设置 Bucket
    bucket_name = 'dp-filetrans-bj'
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    return bucket


class FileSample(BaseModel):
    file_name: Optional[str]
    file_link: Optional[str]
    question: Optional[str]
    answerfile_name: Optional[str]
    answerfile_link: Optional[str]
    compare_fields: List[str]


def get_dataset(data_jsonl: str) -> list[FileSample]:
    bucket = init_oss()
    raw_samples = evals.get_jsonl(data_jsonl)

    for raw_sample in raw_samples:
        if "file_name" in raw_sample:
            oss_file = "changjunhan/" + os.path.basename(raw_sample["file_name"])
            raw_sample["file_link"] = "https://dp-filetrans-bj.oss-cn-beijing.aliyuncs.com/" + oss_file

            exists = bucket.object_exists(oss_file)
            if exists:
                print(f"文件 {oss_file} 已存在于 OSS 中。")
            else:
                # 上传文件
                bucket.put_object_from_file(oss_file, raw_sample["file_name"])
                print(f"文件 {oss_file} 已上传到 OSS。")
        elif "file_link" in raw_sample:
            local_file = raw_sample["file_name"] if "file_name" in raw_sample else os.path.basename(
                raw_sample["file_link"])
            oss_file = "changjunhan/" + os.path.basename(raw_sample["file_link"])
            if not os.path.exists(local_file):
                if bucket.object_exists(oss_file):
                    # 从 OSS 下载文件
                    bucket.get_object_to_file(oss_file, local_file)

    samples = [FileSample(**raw_sample) for raw_sample in raw_samples]
    return samples


class TableExtract(evals.Eval):
    def __init__(
            self,
            completion_fns: list[CompletionFn],
            dataset: str,
            *args,
            instructions: Optional[str] = "",
            **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) < 3, "TableExtract only supports 3 completion fns"
        self.dataset = dataset
        self.instructions = instructions

    def eval_sample(self, sample, rng):
        assert isinstance(sample, FileSample)

        prompt = (
                self.instructions
                + "\nPlease answer in json format."
                + f"\nThe fields should at least contain {sample.compare_fields}"
        )
        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,
            max_tokens=5,
            file_name=sample.file_name,
        )
        sampled = result.get_completions()[0]

        if "csv" in prompt:
            code = re.search(code_pattern, sampled).group()
            code_content = re.sub(code_pattern, r"\1", code)
            table = pd.read_csv(StringIO(code_content))
        elif "json" in prompt:
            code = re.search(code_pattern, sampled).group()
            code_content = re.sub(code_pattern, r"\1", code)
            table = pd.DataFrame(json.loads(code_content))
        else:
            table = pd.DataFrame()
        table = parse_table_multiindex(table)

        correct_answer = pd.read_csv(sample.answerfile)

        for field in sample.compare_fields:
            match_field = field in table.columns and field in correct_answer.columns
            record_match(
                correct=match_field,
                expected=field,
                picked=str(list(table.columns)),
                file_name=sample.file_name,
                jobtype="match_field"
            )
            if match_field:
                match_number = table[field].shape[0] == correct_answer[field].shape[0]
                record_match(
                    correct=match_number,
                    expected=correct_answer[field].shape[0],
                    picked=table[field].shape[0],
                    file_name=sample.file_name,
                    jobtype="match_number"
                )

                for sample_value, correct_value in zip(table[field], correct_answer[field]):
                    record_match(
                        correct=(sample_value == correct_value),
                        expected=correct_value,
                        picked=sample_value,
                        file_name=sample.file_name,
                        jobtype="match_value"
                    )

    def run(self, recorder: RecorderBase):
        samples = get_dataset(self.dataset)
        self.eval_all_samples(recorder, samples)
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }
