import os

import jiant.utils.python.io as py_io
import argparse
from make_task_config import create_task_config


def create_cross_config(args):
    treatments = args.treatments.split(',')

    configs = {}
    for treatment in treatments:
        configs[f'{treatment}_{args.round}'] = create_task_config(
            args,
            write=False,
            data_path=os.path.join(args.data_base, f'{treatment}_{args.round}'),
            itereval=True,
            anlieval=False,
        )

        configs[f'{treatment}_{args.round}_separate'] = create_task_config(
            args,
            write=False,
            data_path=os.path.join(args.data_base, f'{treatment}_{args.round}_separate'),
            itereval=True,
            anlieval=False,
        )

    tasks_dir = os.path.dirname(args.data_base)
    config_dir = os.path.join(tasks_dir, 'configs') if args.config_dir == '' else args.config_dir
    os.makedirs(config_dir, exist_ok=True)

    for training, train_config in configs.items():
        for val in treatments:
            if args.debug and val in training:
                continue

            val_config = configs[f'{val}_{args.round}']

            config_name = f'{training}-{val}_{args.round}' if args.config_name == '' else args.config_name
            config_name = config_name + "_hyp" if args.hypothesis else config_name

            py_io.write_json(
                data={
                    "task": "mnli_hyp" if args.hypothesis else "mnli",
                    "paths": {
                        "train": train_config["paths"]["train"],
                        "val": val_config["paths"]["val"]
                    },
                    "name": "mnli_hyp" if args.hypothesis else "mnli",
                },
                path=os.path.join(config_dir, f'{config_name}_config.json'),
            )


def main():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument('--data_base', type=str, required=True)
    parser.add_argument('--round', type=int, required=True)

    # Optional
    parser.add_argument('--treatments', type=str, default='baseline,LotS,LitL')
    parser.add_argument('--hypothesis', action='store_true')
    parser.add_argument('--config_dir', type=str, default='')
    parser.add_argument('--config_name', type=str, default='')
    parser.add_argument('--task_name', type=str, default='')
    parser.add_argument('--debug', action='store_false')

    args = parser.parse_args()
    create_cross_config(args)


if __name__ == "__main__":
    main()
