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
from jiant.utils.python.io import read_json_lines, read_json, write_json, write_jsonl
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
            new_example["label"] = example["label"]

        # replace pronoun with '_'
        pron_start, pron_end = new_example["pronoun_char_span"]
        new_example["new_text"] = new_example["text"][:pron_start] + '_' + new_example["text"][pron_end+1:]

        # remove duplicate candidates
        new_example["cand_text_list"] = list(set(new_example["cand_text_list"]))

        return new_example

    # write new examples
    expanded_paths = {}
    for split, filename in paths_dict.items():
        examples = read_json_lines(filename)

        # clean up examples and mine candidates
        expanded_examples = list(map(convert_wsc_example, examples))
        cand_lengths = list(map(lambda x: len(x['cand_text_list']), expanded_examples))
        longest = max(cand_lengths)+1

        split_path = os.path.join(data_dir, f'{split}_expanded.jsonl')
        write_jsonl(
            data=expanded_examples,
            path=split_path
        )

        expanded_paths[split] = split_path
    expanded_paths['max_candidates'] = longest

    # write new config
    expanded_config = {key: value for key, value in task_config.items()}
    expanded_config["paths"] = expanded_paths
    write_json(
        data=expanded_config,
        path=os.path.join(config_dir, f'{args.task_name}_mc_config.json')
    )


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
