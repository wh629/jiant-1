import numpy as np
import torch
from dataclasses import dataclass
from typing import List

from jiant.tasks.core import (
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
    Task,
    TaskTypes,
)
from jiant.tasks.lib.templates.shared import single_sentence_featurize, labels_to_bimap
from jiant.utils.python.io import read_jsonl


@dataclass
class Example(BaseExample):
    guid: str
    premise: str
    hypothesis: str
    label: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            hypothesis=tokenizer.tokenize(self.hypothesis),
            label_id=MnliHypTask.LABEL_TO_ID[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    hypothesis: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        return single_sentence_featurize(
            guid=self.guid,
            input_tokens_b=self.hypothesis,
            label_id=self.label_id,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            data_row_class=DataRow,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label_id: int
    tokens: list


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.LongTensor
    tokens: list


class MnliHypTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.CLASSIFICATION
    LABELS = ["contradiction", "entailment", "neutral"]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    def get_train_examples(self):
        return self._create_examples(lines=read_jsonl(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_jsonl(self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_jsonl(self.test_path), set_type="test")

    @classmethod
    def _create_examples(cls, lines, set_type):
        # noinspection DuplicatedCode
        examples = []
        for (i, line) in enumerate(lines):
            examples.append(
                Example(
                    # NOTE: get_glue_preds() is dependent on this guid format.
                    guid="%s-%s" % (set_type, i),
                    hypothesis=line["hypothesis"],
                    label=line["label"] if set_type != "test" else cls.LABELS[-1],
                )
            )
        return examples
