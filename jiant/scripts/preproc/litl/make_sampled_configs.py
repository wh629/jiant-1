import os

import jiant.utils.python.io as py_io
import argparse
from make_task_config import create_task_config


def create_sampled_config(args):
    treatments = args.treatments.split(',')

    configs = {}
    for treatment in treatments:
        configs[f'{treatment}_{args.round}'] = create_task_config(
            args,
            write=False,
            data_path=os.path.join(args.data_base, f'{treatment}_{args.round}'),
            itereval=True,
            mnlieval=False,
        )

        configs[f'{treatment}_{args.round}_separate'] = create_task_config(
            args,
            write=False,
            data_path=os.path.join(args.data_base, f'{treatment}_{args.round}_separate'),
            itereval=True,
            mnlieval=False,
        )

    tasks_dir = os.path.dirname(args.data_base)
    config_dir = os.path.join(tasks_dir, 'configs', args.sample) if args.config_dir == '' else args.config_dir
    os.makedirs(config_dir, exist_ok=True)

    for training, train_config in configs.items():
        train_base_dir = os.path.join(args.data_base, training, args.sample)

        for split in next(os.walk(train_base_dir))[1]:
            config_name = f'{training}' if args.config_name == '' else args.config_name
            config_name = config_name + "_hyp" if args.hypothesis else config_name

            out_dir = os.path.join(config_dir, split)

            os.makedirs(out_dir, exist_ok=True)

            train_dir, train_f = os.path.split(train_config["paths"]["train"])
            train_path = os.path.join(train_dir, args.sample, split, train_f)

            # In-domain Val Data
            if not args.no_indomain:
                py_io.write_json(
                    data={
                        "task": "mnli_hyp" if args.hypothesis else "mnli",
                        "paths": {
                            "train": train_path,
                            "val": train_config["paths"]["val"]
                        },
                        "name": "mnli_hyp" if args.hypothesis else "mnli",
                    },
                    path=os.path.join(out_dir, f'{config_name}_config.json'),
                )

            if not args.hypothesis:
                if not args.itereval_path == '':
                    # Itereval
                    py_io.write_json(
                        data={
                            "task": "mnli",
                            "paths": {
                                "train": train_path,
                                "val": args.itereval_path
                            },
                            "name": "mnli",
                        },
                        path=os.path.join(out_dir, f'eval_{config_name}_config.json'),
                    )

                if not args.eval_paths == '':
                    assert len(args.eval_paths.split(',')) == len(args.eval_names.split(','))

                    for eval_path, eval_name in zip(args.eval_paths.split(','), args.eval_names.split(',')):
                        # eval
                        py_io.write_json(
                            data={
                                "task": "mnli",
                                "paths": {
                                    "train": train_path,
                                    "val": eval_path
                                },
                                "name": "mnli",
                            },
                            path=os.path.join(out_dir, f'{eval_name}_{config_name}_config.json'),
                        )


def main():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument('--data_base', type=str, required=True)
    parser.add_argument('--round', type=int, required=True)

    # Optional
    parser.add_argument('--no_indomain', action='store_true')
    parser.add_argument('--itereval_path', type=str, default='')
    parser.add_argument('--eval_paths', type=str, default='')
    parser.add_argument('--eval_names', type=str, default='')


    parser.add_argument('--sample', type=str, default='cross_eval')
    parser.add_argument('--treatments', type=str, default='baseline,LotS,LitL')
    parser.add_argument('--hypothesis', action='store_true')
    parser.add_argument('--config_dir', type=str, default='')
    parser.add_argument('--config_name', type=str, default='')
    parser.add_argument('--task_name', type=str, default='')

    args = parser.parse_args()
    create_sampled_config(args)


if __name__ == "__main__":
    main()
