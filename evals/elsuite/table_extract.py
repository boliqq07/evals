from io import StringIO
import json
import os
import re

from typing import List, Optional, Tuple, Union

import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider

import pandas as pd
from pydantic import BaseModel

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.record import RecorderBase, record_match

code_pattern = r"```[\s\S]*?\n([\s\S]+?)\n```"
json_pattern = r"```json[\s\S]*?\n([\s\S]+?)\n```"
csv_pattern = r"```csv[\s\S]*?\n([\s\S]+?)\n```"


def parse_table_multiindex(table: pd.DataFrame) -> pd.DataFrame:
    """
    Parse a table with multiindex columns.
    """

    df = table.copy()
    if df.columns.nlevels == 1:
        coltypes = {col: type(df[col].iloc[0]) for col in df.columns}
        for col, ctype in coltypes.items():
            if ctype == str:
                if ":" in df[col].iloc[0] and "," in df[col].iloc[0]:
                    df[col] = [{key: value for key, value in [pair.split(": ") for pair in data.split(", ")]} for data
                               in df[col]]
                    coltypes[col] = dict
        dfs = []

        for col, ctype in coltypes.items():
            if ctype == dict:
                d = pd.DataFrame(df.pop(col).tolist())
                d.columns = pd.MultiIndex.from_tuples([(col, fuzzy_normalize(key)) for key in d.columns])
                dfs.append(d)
        df.columns = pd.MultiIndex.from_tuples([(col, "") for col in df.columns])
        df = pd.concat([df] + dfs, axis=1)
    if df.columns.nlevels > 1:
        df.columns = pd.MultiIndex.from_tuples([(col, fuzzy_normalize(subcol)) for col, subcol in df.columns])

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
    answerfile_name: Optional[str]
    answerfile_link: Optional[str]
    compare_fields: List[Union[str, Tuple]]


def get_dataset(data_jsonl: str) -> list[FileSample]:
    bucket = init_oss()
    raw_samples = evals.get_jsonl(data_jsonl)

    for raw_sample in raw_samples:
        for ftype in ["", "answer"]:
            if f"{ftype}file_name" in raw_sample:
                oss_file = "changjunhan/" + os.path.basename(raw_sample[f"{ftype}file_name"])
                raw_sample[f"{ftype}file_link"] = "https://dp-filetrans-bj.oss-cn-beijing.aliyuncs.com/" + oss_file

                exists = bucket.object_exists(oss_file)
                if exists:
                    print(f"文件 {oss_file} 已存在于 OSS 中。")
                else:
                    # 上传文件
                    bucket.put_object_from_file(oss_file, raw_sample[f"{ftype}file_name"])
                    print(f"文件 {oss_file} 已上传到 OSS。")
            elif f"{ftype}file_link" in raw_sample:
                local_file = raw_sample[f"{ftype}file_name"] if f"{ftype}file_name" in raw_sample else os.path.basename(
                    raw_sample[f"{ftype}file_link"])
                oss_file = "changjunhan/" + os.path.basename(raw_sample[f"{ftype}file_link"])
                if not os.path.exists(local_file):
                    if bucket.object_exists(oss_file):
                        # 从 OSS 下载文件
                        bucket.get_object_to_file(oss_file, local_file)
        raw_sample["compare_fields"] = [field if type(field) == str else tuple(field) for field in
                                        raw_sample["compare_fields"]]
    print(raw_samples)
    samples = [FileSample(**raw_sample) for raw_sample in raw_samples]
    return samples


def fuzzy_compare(a: str, b: str) -> bool:
    """
    Compare two strings with fuzzy matching.
    """

    def standardize_unit(s: str) -> str:
        """
        Standardize a (affinity) string to common units.
        """
        mark = "" if re.search(r"[><=]", s) is None else re.search(r"[><=]", s).group()
        unit = s.rstrip()[-2:]
        number = re.search(r"[0-9.\+\-]+", s).group()

        if unit in ["µM", "uM"]:
            unit = "nM"
            number *= 1000
        elif unit in ["mM", "mm"]:
            unit = "nM"
            number *= 1000000

        if mark == "=":
            mark = ""
        return f"{mark}{number:.1f} {unit}"

    unit_str = ["nM", "uM", "µM", "mM"]
    a = a.strip()
    b = b.strip()
    if a[-2:] in unit_str and b[-2:] in unit_str:
        a = standardize_unit(a)
        b = standardize_unit(b)
        return a == b
    else:
        return (a.lower() in b.lower()) or (b.lower() in a.lower())


def fuzzy_normalize(s):
    """ 标准化字符串 """
    # 定义需要移除的单位和符号
    units = ["µM", "µg/mL", "nM"]
    for unit in units:
        s = s.replace(unit, "")

    # 定义特定关键字
    keywords = ["IC50", "EC50", "TC50", "GI50", "Ki", "Kd"]

    # 移除非字母数字的字符，除了空格
    s = re.sub(r'[^\w\s]', '', s)

    # 分割字符串为单词列表
    words = s.split()

    # 将关键字移到末尾
    reordered_words = [word for word in words if word not in keywords]
    keywords_in_string = [word for word in words if word in keywords]
    reordered_words.extend(keywords_in_string)

    # 重新组合为字符串
    return ' '.join(reordered_words)


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
                + f"\nThe fields should at least contain {sample.compare_fields}"
        )
        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,
            max_tokens=5,
            file_name=sample.file_name,
            file_link=sample.file_link
        )
        sampled = result.get_completions()[0]

        if "csv" in prompt:
            code = re.search(code_pattern, sampled).group()
            code_content = re.sub(code_pattern, r"\1", code)
            table = pd.read_csv(StringIO(code_content))
            if pd.isna(table.iloc[0, 0]):
                table = pd.read_csv(StringIO(code_content), header=[0, 1])

        elif "json" in prompt:
            code = re.search(code_pattern, sampled).group()
            code_content = re.sub(code_pattern, r"\1", code)
            table = pd.DataFrame(json.loads(code_content))
        else:
            table = pd.DataFrame()
        table = parse_table_multiindex(table)
        table.to_csv("temp1.csv")

        correct_answer = parse_table_multiindex(pd.read_csv(sample.answerfile_name, header=[0, 1]).astype(str))

        for field in sample.compare_fields:
            if type(field) == tuple:
                field = (field[0], fuzzy_normalize(field[1]))
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
                        correct=fuzzy_compare(sample_value, correct_value),
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
