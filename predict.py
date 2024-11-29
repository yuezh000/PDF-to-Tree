import argparse
import os
import sys
import json
import logging
import traceback as tb

root_dir = os.path.realpath(os.path.dirname(__file__))
p2t_dir = os.path.join(root_dir, "p2t")

if p2t_dir not in sys.path:
    sys.path.insert(0, p2t_dir)
                       

from model_entry import ModelEntry, e2e_predict
from utils import load_jsonl, load_pickle



def predict(workdir, data_dir):
    model_entry = ModelEntry(workdir=workdir)
    feature_extractor = model_entry.model.feature_extractor

    test_data = load_jsonl(os.path.join(data_dir, f"test.jsonl"))
    if feature_extractor.enc_type in {"layoutlmv2", "layoutlmv3"}:
        test_images = load_pickle(os.path.join(data_dir, f"test_images.pkl"))
        for item in test_data:
            item['images'] = test_images[item['doc_id']]

    _, parser_metrics = e2e_predict(model_entry=model_entry, workdir=workdir, inp_data=test_data, output_name="test")
    logging.info(json.dumps(parser_metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--workdir", type=str)
    parser.add_argument("-d", "--data_dir", type=str)
    parser.add_argument("-c", "--config_file", type=str, default=None)

    args = parser.parse_args()

    try:
        if args.config_file is not None:
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

        predict(args.workdir, args.data_dir)

    except ValueError as e:
        logging.error(e)
        logging.error(tb.format_exc())
        raise e


if __name__ == '__main__':
    main()
