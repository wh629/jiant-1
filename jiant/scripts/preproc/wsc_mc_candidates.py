import os
import pickle
import argparse
import json
import jiant.utils.zconf as zconf
from jiant.scripts.preproc.fairseq_wsc.wsc_utils import (
    filter_noun_chunks,
    extended_noun_chunks,
    get_detokenizer,
    get_spacy_nlp,
)
from jiant.utils.python.io import read_json_lines, read_json
import torch
import pandas as pd


def strip_punc(x):
    return x.rstrip('.,;"').lstrip('"')


def find_first_char_span(text, target, starting_idx=0):
    span = (starting_idx, min(starting_idx + len(target), len(text)))
    for idx in range(starting_idx, len(text) - len(target)):
        if text[idx : idx + len(target)].lower() == target.lower():
            span = (idx, idx + len(target))
            break
    return text[slice(*span)], span


def process_wsc_candidates(args):
    task_config = read_json(args.task_config)
    paths_dict = task_config["paths"]
    config_dir, data_dir = os.path.dirname(args.task_config), os.path.dirname(paths_dict["train"])
    detok = get_detokenizer()
    nlp = get_spacy_nlp()

    # single instance process
    # cleanup weird chars, realign spans
    def convert_wsc_example(example):
        new_example = {}
        new_example["uid"] = f'{split}_{example["idx"]}'
        tokens = example["text"].replace("\n", " ").split()
        new_example["text"] = detok.detokenize(tokens)
        new_example["query_text"] = strip_punc(
            example["target"]["span1_text"].replace("\n", " ").lower()
        )
        new_example["query_text"], new_example["query_char_span"] = find_first_char_span(
            text=new_example["text"],
            target=new_example["query_text"],
            starting_idx=len(detok.detokenize(tokens[: example["target"]["span1_index"]])),
        )
        new_example["pronoun_text"] = strip_punc(
            example["target"]["span2_text"].replace("\n", " ").lower()
        )
        new_example["pronoun_text"], new_example["pronoun_char_span"] = find_first_char_span(
            text=new_example["text"],
            target=new_example["pronoun_text"],
            starting_idx=len(detok.detokenize(tokens[: example["target"]["span2_index"]])),
        )

        # get candidates
        new_example["cand_text_list"] = [
            cand.text
            for cand in filter_noun_chunks(
                extended_noun_chunks(nlp(new_example["text"])),
                exclude_pronouns=True,
                exclude_query=new_example["query_text"],
                exact_match=False,
            )
        ]

        if split != "test":
            new_example["p_label"] = example["label"]
        return new_example

    for split, filename in paths_dict.items():
        examples = read_json_lines(filename)

        # clean up examples and mine candidates
        expanded_examples = list(map(convert_wsc_example, examples))

        with open(os.path.join(data_dir, f'{split}_expanded.jsonl'), w) as f:
            for example in expanded_examples:
                f.write(f'{json.dumps(example)}\n')

        # # cross example process
        # global_ans_dict = {}
        # for idx, example in enumerate(examples):
        #     key = (example["text"], example["pronoun_text"])
        #     if key not in global_ans_dict:
        #         global_ans_dict[key] = {"correct_query": None, "idxs": [], "all_cands": []}
        #     global_ans_dict[key]["idxs"].append(idx)
        #     global_ans_dict[key]["all_cands"].append(example["query_text"])
        #     if example.get("p_label", False):
        #         global_ans_dict[key]["correct_query"] = example["query_text"]
        #
        # for example_group in global_ans_dict.values():
        #     correct_query = example_group["correct_query"]
        #     for idx in example_group["idxs"]:
        #         example = examples[idx]
        #         # if candidates_source == "cross":
        #         #     example["cand_text_list"] = [
        #         #         cand
        #         #         for cand in example_group["all_cands"]
        #         #         if cand != example["query_text"]
        #         #     ]
        #         example["cand_text_list"] = list(set(example["cand_text_list"]))
        #         if split == "train":
        #             if correct_query is not None:
        #                 query_and_cands = [example["query_text"]] + example["cand_text_list"]
        #                 try:
        #                     example["mc_label"] = query_and_cands.index(correct_query)
        #                 except ValueError:
        #                     example["cand_text_list"].insert(0, correct_query)
        #                     example["mc_label"] = 1
        #             else:
        #                 example["mc_label"] = -1
        # return examples

        # self.raw_data = {
        #     "train": load_wsc_split(os.path.join(self.data_dir, "WSC", "train.jsonl"), "train"),
        #     "val": load_wsc_split(os.path.join(self.data_dir, "WSC", "val.jsonl"), "val"),
        #     "test": load_wsc_split(os.path.join(self.data_dir, "WSC", "test.jsonl"), "test"),
        # }


def main():
    parser = argparse.ArgumentParser()
    # === Required parameters === #
    parser.add_argument('--task_config', type=str, required=True, help='original task config dict')

    # === Optional parameters === #
    parser.add_argument('--task_name', type=str, default="wsc", help='name of task')

    args = parser.parse_args()

    process_wsc_candidates(args)


if __name__ == "__main__":
    main()
