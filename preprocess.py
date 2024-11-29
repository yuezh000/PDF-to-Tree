import os
import sys
import shutil
import multiprocessing
import pickle
import torch
from fastNLP import DataSet
from fastNLP import logger as logging
import pandas as pd
from argparse import ArgumentParser

root_dir = os.path.realpath(os.path.dirname(__file__))
p2t_dir = os.path.join(root_dir, "p2t")

if p2t_dir not in sys.path:
    sys.path.insert(0, p2t_dir)

from parser.data import process_data, DocLayoutLoader, DEFAULT_STACK_WINDOW_SIZE, DEFAULT_BUF_WINDOW_SIZE
from utils import load_jsonl, save_jsonl, load_json


def tree_to_action(args):
    # Data Processing
    assert args.input != args.output, args.input

    shutil.copy(os.path.join(args.input, "labels.json"), os.path.join(args.output, "labels.json"))
    tree_data = load_jsonl(os.path.join(args.input, "test.jsonl"))
    save_jsonl(os.path.join(args.output, "test.jsonl"), tree_data)

    data_loader = DocLayoutLoader(stack_win_size=args.stack_win_size, buffer_win_size=args.buffer_win_size)
    logging.info(f"Loading {args.input}/[train|test|dev].jsonl")
    data_bundle = data_loader.load({
        "train": os.path.join(args.input, "train.jsonl"),
        "dev": os.path.join(args.input, "dev.jsonl"),
        "test": os.path.join(args.input, "test.jsonl"),
    })

    num_proc = int(multiprocessing.cpu_count() / 2)

    node_labels = load_json(os.path.join(args.input, "labels.json"))

    data_bundle, tokenizer, test_images = process_data(
        data_bundle, node_labels=node_labels, 
        image_dir=args.image_dir, model_name_or_path=args.model_name_or_path, num_proc=num_proc,
        use_rel_pos=args.use_rel_pos, use_stack_label=args.use_stack_label, use_ptr=args.use_ptr,
        stack_win_size=args.stack_win_size, buffer_win_size=args.buffer_win_size, use_auto_trunc=args.use_auto_trunc,
        use_font_size=args.use_font_size, use_wh=args.use_wh
    )

    for ds_name in ("train", "dev", "test"):
        out_path = os.path.join(args.output, f"{ds_name}.pkl")
        logging.info(f"Saving {out_path}")
        data_bundle.get_dataset(ds_name).save(os.path.join(args.output, f"{ds_name}.pkl"))

    if test_images is not None:
        out_path = os.path.join(args.output, f"test_images.pkl")
        logging.info(f"Saving {out_path}")
        with open(out_path, 'wb') as out_file:
            pickle.dump(test_images, out_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-img", "--image_dir", type=str)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-m", "--model_name_or_path", type=str)
    parser.add_argument("--use_rel_pos", action="store_true")
    parser.add_argument("--use_stack_label", action="store_true")
    parser.add_argument("--stack_win_size", type=int, default=DEFAULT_STACK_WINDOW_SIZE)
    parser.add_argument("--buffer_win_size", type=int, default=DEFAULT_BUF_WINDOW_SIZE)
    parser.add_argument("--use_ptr", action="store_true")
    parser.add_argument("--use_auto_trunc", action="store_true")
    parser.add_argument("--use_font_size", action="store_true")
    parser.add_argument("--use_wh", action="store_true")

    args = parser.parse_args()

    if args.config is not None:
        config = load_json(args.config)
        args.input = config.get('anno_dir', args.input)
        args.output = config.get('data_dir', args.output)
        args.image_dir = config.get('image_dir', args.image_dir)
        for hyper in config['hypers']:
            if hyper['name'] == "model_name_or_path":
                args.model_name_or_path = hyper['value']
            if hyper['name'] == "use_rel_pos":
                args.use_rel_pos = hyper['value']
            if hyper['name'] == "use_stack_label":
                args.use_stack_label = hyper['value']
            if hyper['name'] == "stack_win_size":
                args.stack_win_size = hyper['value']
            if hyper['name'] == "buffer_win_size":
                args.buffer_win_size = hyper['value']
            if hyper['name'] == "use_ptr":
                args.use_ptr = hyper['value']
            if hyper['name'] == "use_auto_trunc":
                args.use_auto_trunc = hyper['value']
            if hyper['name'] == "use_font_size":
                args.use_font_size = hyper['value']
            if hyper['name'] == "use_wh":
                args.use_wh = hyper['value']

    args.input = os.path.realpath(args.input)
    args.output = os.path.realpath(args.output)
    args.image_dir = os.path.realpath(args.image_dir)

    tree_to_action(args)

