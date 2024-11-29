import os
import sys
import json
import shutil
import logging
import multiprocessing
import pickle
import traceback as tb
from argparse import ArgumentParser

root_dir = os.path.realpath(os.path.dirname(__file__))
p2t_dir = os.path.join(root_dir, "p2t")

if p2t_dir not in sys.path:
    sys.path.insert(0, p2t_dir)


from model_entry import ModelEntry


def train():
    parser = ArgumentParser()
    parser.add_argument("-w", "--workdir", type=str)
    parser.add_argument("-d", "--data_dir", type=str)
    parser.add_argument("-c", "--config_file", type=str)

    args = parser.parse_args()

    try:
        if not os.path.isfile(args.config_file):
            raise ValueError(f"Config file not found: {args.config_file}")

        with open(args.config_file) as config_file:
            config = json.loads(config_file.read())

        args.workdir = config.get('workdir', args.workdir)
        args.data_dir = config.get('data_dir', args.data_dir)

        if not os.path.isdir(args.workdir):
            raise ValueError(f"Work dir not found: {args.workdir}")

        if not os.path.isdir(args.data_dir):
            raise ValueError(f"Data dir not found: {args.data_dir}")

        model_entry = ModelEntry()
        model_entry.fit(workdir=args.workdir, data_dir=args.data_dir, config=config)
    except ValueError as e:
        logging.error(e)
        logging.error(tb.format_exc())
        raise e


if __name__ == '__main__':
    train()