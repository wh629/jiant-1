import os

import jiant.utils.python.io as py_io
import argparse


def create_task_config(args, write=True, data_path=None, itereval=True, mnlieval=True):
    if not data_path is None:
        args.data_path = data_path

    files = []
    for (dirpath, dirnames, filenames) in os.walk(args.data_path):
        files.extend(filenames)
        break

    paths = {}
    for file in files:
        split = file.split('_')[0]
        paths[split] = os.path.join(args.data_path, file)

    tasks_dir = os.path.dirname(os.path.dirname(args.data_path))
    config_dir = os.path.join(tasks_dir, 'configs') if args.config_dir == '' else args.config_dir
    os.makedirs(config_dir, exist_ok=True)

    config_name = os.path.basename(args.data_path) if args.config_name == '' else args.config_name
    config_name = config_name + "_hyp" if args.hypothesis else config_name

    print(config_name)

    if write:
        py_io.write_json(
            data={
                "task": "mnli_hyp" if args.hypothesis else "mnli",
                "paths": paths,
                "name": "mnli_hyp" if args.hypothesis else "mnli",
            },
            path=os.path.join(config_dir, f'{config_name}_config.json'),
        )

        if not args.hypothesis:
            if itereval and args.itereval_path != '':
                py_io.write_json(
                    data={
                        "task": "mnli" if args.task_name == '' else args.task_name,
                        "paths": {
                            "train": paths["train"],
                            "val": args.itereval_path
                        },
                        "name": "mnli",
                    },
                    path=os.path.join(config_dir, f'eval_{config_name}_config.json'),
                )
            if mnlievals and args.eval_paths != '':
                for eval_path, eval_name in zip(args.eval_paths.split(','), args.eval_names.split(',')):
                    py_io.write_json(
                        data={
                            "task": "mnli" if args.task_name == '' else args.task_name,
                            "paths": {
                                "train": paths["train"],
                                "val": eval_path
                            },
                            "name": "mnli",
                        },
                        path=os.path.join(config_dir, f'{eval_name}_{config_name}_config.json'),
                    )
    else:
        return {
                "task": "mnli_hyp" if args.hypothesis else "mnli",
                "paths": paths,
                "name": "mnli_hyp" if args.hypothesis else "mnli",
            }


def main():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument('--data_path', type=str, required=True)

    # Optional
    parser.add_argument('--itereval_path', type=str, default='')
    parser.add_argument('--eval_paths', type=str, default='')
    parser.add_argument('--eval_names', type=str, default='')

    parser.add_argument('--hypothesis', action='store_true')
    parser.add_argument('--config_dir', type=str, default='')
    parser.add_argument('--config_name', type=str, default='')
    parser.add_argument('--task_name', type=str, default='')

    args = parser.parse_args()
    create_task_config(args)


if __name__ == "__main__":
    main()
