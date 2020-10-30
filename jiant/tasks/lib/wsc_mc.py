from dataclasses import dataclass

import pandas as pd

from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import multiple_choice as mc_template
from jiant.utils.python.io import read_json_lines


@dataclass
class Example(mc_template.Example):
    @property
    def task(self):
        return WSCMCTask


@dataclass
class TokenizedExample(mc_template.TokenizedExample):
    pass


@dataclass
class DataRow(mc_template.DataRow):
    pass


@dataclass
class Batch(mc_template.Batch):
    pass


class WSCMCTask(mc_template.AbstractMultipleChoiceTask):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    CHOICE_KEYS = [0, 1]
    CHOICE_TO_ID, ID_TO_CHOICE = labels_to_bimap(CHOICE_KEYS)
    NUM_CHOICES = len(CHOICE_KEYS)

    def get_train_examples(self):
        return self._create_examples(self.train_path, self.path_dict["train_max_cand"], set_type="train")

    def get_val_examples(self):
        return self._create_examples(self.val_path, self.path_dict["val_max_cand"], set_type="val")

    def get_test_examples(self):
        return self._create_examples(self.test_path, self.path_dict["test_max_cand"], set_type="test")

    @classmethod
    def _create_examples(cls, path, max_cand, set_type):
        lines = read_json_lines(path)
        examples = []

        for line in lines:
            if set_type == "train" and line['label'] == False:
                continue

            pron_pre, pron_post = line['new_text'].split('_')
            mined_choices = [line['query_text']]+line['cand_text_list']
            choice_list = [cand + pron_post for cand in mined_choices]

            # pad with dummy candidates
            for _ in range(max_cand-len(mined_choices)):
                choice_list.append('.' + pron_post)

            if set_type == 'train':
                label = 0
            elif set_type == 'val':
                label = int(line['label'])
            else:
                label = cls.CHOICE_KEYS[-1]

            examples.append(
                Example(
                    guid=line['uid'],
                    prompt=pron_pre,
                    choice_list=choice_list,
                    label=label,
                )
            )

        return examples
