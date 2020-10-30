import csv
import os
import tqdm

import jiant.utils.python.io as py_io
import jiant.utils.zconf as zconf


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    data_path = zconf.attr(type=str, required=True)
    itereval_path = zconf.attr(type=str, required=True)

    # Optional
    hypothesis = zconf.attr(action='store_true')
    config_dir = zconf.attr(type=str, default='')
    config_name = zconf.attr(type=str, default='')
    task_name = zconf.attr(type=str, default='')


def create_task_config(args):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(args.data_path):
        files.extend(filenames)
        break

    paths = {}
    for file in files:
        split = file.split('_')[0]
        paths[split] = os.path.join(args.data_path, file)

    config_dir = os.path.join('.', 'tasks', 'configs') if args.config_dir == '' else args.config_dir
    os.makedirs(config_dir, exist_ok=True)

    config_name = os.path.basename(args.data_path) if args.config_name == '' else args.config_name
    config_name = config_name + "_hyp" if args.hypothesis else config_name

    py_io.write_json(
        data={
            "task": "mnli" if args.task_name == '' else args.task_name,
            "paths": paths,
            "name": "mnli_hyp" if args.hypothesis else "mnli",
        },
        path=os.path.join(config_dir, f'{config_name}_config.json'),
    )

    if not args.hypothesis:
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


def main():
    args = RunConfiguration.default_run_cli()
    create_task_config(args)


if __name__ == "__main__":
    main()
