import argparse
import os
import json
import shutil
import traceback as tb
import random
import numpy as np
import time
import torch.cuda
import yaml
from torch import optim
from fastNLP import Trainer, LoadBestModelCallback, TorchWarmupCallback
from fastNLP import LRSchedCallback
from fastNLP import Evaluator
from fastNLP import ClassifyFPreRecMetric
from fastNLP import prepare_torch_dataloader, BucketedBatchSampler
from fastNLP import DataSet
from fastNLP import logger as logging
from fastNLP.io import DataBundle
from parser.data import DocLayoutFeatureExtractor
from model import DocLayoutParserModel
from metric import eval_parser_pred, eval_cls_pred, eval_speed
from utils import load_json, save_json, load_jsonl, load_pickle
from parser.state import ParserState
from callback import SaveEvalResultCallback
from torchsampler import ImbalancedDatasetSampler
from torch.optim.lr_scheduler import LinearLR


class ModelEntry(object):
    def __init__(self, workdir: str = None):
        self.model = None
        self.app_root = os.path.realpath(os.path.dirname(__file__))
        try:
            with open(self._root_path("model.yml")) as inp_file:
                self.model_yaml = yaml.safe_load(inp_file)
        except yaml.YAMLError as e:
            logging.error(f"Failed to load {self._root_path('model.yml')}: {e}")

        if workdir is not None:
            self._load_model(workdir)

    def _root_path(self, path):
        return os.path.join(self.app_root, path)

    def fit(self, workdir, data_dir, config):
        # Handle hyper parameters
        model_hyper = dict([(h["name"], h["default_value"]) for h in self.model_yaml["hypers"]])

        for hyper in config["hypers"]:
            model_hyper[hyper['name']] = hyper['value']

        labels_file_path = os.path.join(data_dir, "labels.json")
        if not os.path.isfile(labels_file_path):
            raise ValueError(f"File not found: {labels_file_path}")
        shutil.copy(labels_file_path, os.path.join(workdir, "labels.json"))

        node_labels = load_json(labels_file_path)
        model_name_or_path = model_hyper['model_name_or_path']
        lr = model_hyper['learning_rate']
        n_epochs = model_hyper['n_epochs']
        batch_size = model_hyper['batch_size']
        if model_hyper['random_seed'] is None:
            model_hyper['random_seed'] = random.randint(0, 99999)
        else:
            model_hyper['random_seed'] = int(model_hyper['random_seed'])

        logging.info("Hyper parameters:")
        for k, v in model_hyper.items():
            logging.info(f"  {k} = {v}")

        save_json(os.path.join(workdir, "hypers.json"), model_hyper)

        self.set_seed(model_hyper['random_seed'])

        # Data Processing
        data_bundle = DataBundle(datasets={
            "train": DataSet.load(os.path.join(data_dir, "train.pkl")),
            "dev": DataSet.load(os.path.join(data_dir, "dev.pkl")),
            "test": DataSet.load(os.path.join(data_dir, "test.pkl")),
        })

        logging.info(f"Train samples: {len(data_bundle.get_dataset('train'))}")
        logging.info(f"Dev samples: {len(data_bundle.get_dataset('dev'))}")
        logging.info(f"Test samples: {len(data_bundle.get_dataset('test'))}")


        train_sampler = ImbalancedDatasetSampler(data_bundle.datasets['train'],
                                         callback_get_label=lambda x: x['raw_target'])

        dls = {
            "train": prepare_torch_dataloader(data_bundle.datasets['train'], 
                                              sampler=train_sampler,
                                              batch_size=batch_size, shuffle=False),
            "dev": prepare_torch_dataloader(data_bundle.datasets['dev'], batch_size=batch_size),
            "test": prepare_torch_dataloader(data_bundle.datasets['test'], batch_size=batch_size),
        }

        feature_extractor = DocLayoutFeatureExtractor(
            model_name_or_path, node_labels=node_labels,
            use_stack_label=model_hyper['use_stack_label'],
            use_rel_pos=model_hyper['use_rel_pos'],
            stack_win_size=model_hyper['stack_win_size'],
            buffer_win_size=model_hyper['buffer_win_size'],
            use_ptr=model_hyper['use_ptr'],
            use_auto_trunc=model_hyper['use_auto_trunc'],
            use_font_size=model_hyper['use_font_size'],
            use_wh=model_hyper['use_wh'],
        )

        # Initialize model, trainer and evaluator and run
        model = DocLayoutParserModel(model_name_or_path=model_name_or_path,
                                     feature_extractor=feature_extractor, use_ptr=model_hyper['use_ptr'],
                                     alpha=model_hyper['alpha'], beta=model_hyper["beta"],
                                     dropout=model_hyper['dropout'])

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        callbacks = [
            LoadBestModelCallback(),  # 用于在训练结束之后加载性能最好的model的权重
            TorchWarmupCallback(),
            SaveEvalResultCallback(workdir),
            LRSchedCallback(LinearLR(optimizer, verbose=True))
        ]

        device = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else "cpu"
        if len(device) == 1:
            device = device[0]
        logging.info(f"Using device: {device}")
        overfit_batches = 0

        print(data_bundle.get_dataset('train'))

        trainer = Trainer(model=model, train_dataloader=dls['train'], optimizers=optimizer,
                          evaluate_dataloaders=dls['dev'],
                          train_input_mapping=model.train_input_mapping,
                          evaluate_input_mapping=model.evaluate_input_mapping,
                          metrics={'f': ClassifyFPreRecMetric(tag_vocab=feature_extractor.target_vocab, only_gross=False)},
                          n_epochs=n_epochs, callbacks=callbacks,
                          torch_kwargs={'ddp_kwargs':{'find_unused_parameters': True}},
                          accumulation_steps=model_hyper['accumulation_steps'],
                          device=device, monitor='f#f', fp16=False, overfit_batches=overfit_batches)
        trainer.run()

        evaluator = Evaluator(
            model=model,
            driver=trainer.driver,  # 需要使用 trainer 已经启动的 driver
            device=None,
            dataloaders=dls['test'] if overfit_batches == 0 else trainer.dataloader,
            metrics={'f': ClassifyFPreRecMetric(tag_vocab=feature_extractor.target_vocab, only_gross=False)}
        )

        result = evaluator.run()

        eval_result = {
            "key_metric": "f1-score",
            "metrics": [
                {"name": "f1-score", "value": result['f#f']},
                {"name": "precision", "value": result['pre#f']},
                {"name": "recall", "value": result['rec#f']}
            ]
        }

        # Save result
        save_json(os.path.join(workdir, "eval_result.final.json"), eval_result)

        # Save model
        model = trainer.model
        torch.save(model, os.path.join(workdir, "best_model.pkl"))
        
        model_name_or_path = model_hyper['model_name_or_path']
        model_name = model_name_or_path.rstrip("/").split("/")[-1]
        tokenizer_path = os.path.join(workdir, "tokenizer", model_name)
        model.tokenizer.save_pretrained(tokenizer_path)

        self.model=trainer.model
        
        test_data = load_jsonl(os.path.join(data_dir, f"test.jsonl"))
        if feature_extractor.enc_type in {"layoutlmv2", "layoutlmv3"}:
            test_images = load_pickle(os.path.join(data_dir, f"test_images.pkl"))
            for item in test_data:
                item['images'] = test_images[item['doc_id']]
        e2e_predict(model_entry=self, workdir=workdir, inp_data=test_data, output_name="before-load-test")
        
        debug_predict_cls(model_entry=self, workdir=workdir, ds=data_bundle.get_dataset("test"), output_name_name="test")
        # END

        # Extra Evaluation for debug
        self._load_model(workdir)
        test_data = load_jsonl(os.path.join(data_dir, f"test.jsonl"))
        if feature_extractor.enc_type in {"layoutlmv2", "layoutlmv3"}:
            test_images = load_pickle(os.path.join(data_dir, f"test_images.pkl"))
            for item in test_data:
                item['images'] = test_images[item['doc_id']]
        e2e_predict(model_entry=self, workdir=workdir, inp_data=test_data, output_name="test")

    def set_seed(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _load_model(self, workdir):
        labels_file_path = os.path.join(workdir, "labels.json")
        if not os.path.isfile(labels_file_path):
            raise ValueError(f"File not found: {labels_file_path}")
        node_labels = load_json(labels_file_path)

        hyper_file_path = os.path.join(workdir, "hypers.json")
        if not os.path.isfile(hyper_file_path):
            raise ValueError(f"File not found: {hyper_file_path}")
        model_hyper = load_json(hyper_file_path)

        model_name_or_path = model_hyper['model_name_or_path']
        model_name = model_name_or_path.rstrip("/").split("/")[-1]
        tokenizer_path = os.path.join(workdir, "tokenizer", model_name)
        if not os.path.isdir(tokenizer_path):
            raise ValueError(f"Dir not found: {tokenizer_path}")
        logging.info(f"Loading tokenizer: {tokenizer_path}")
        feature_extractor = DocLayoutFeatureExtractor(
            tokenizer_path, node_labels=node_labels,
            use_stack_label=model_hyper['use_stack_label'],
            use_rel_pos=model_hyper['use_rel_pos'],
            stack_win_size=model_hyper['stack_win_size'],
            buffer_win_size=model_hyper['buffer_win_size'],
            use_ptr=model_hyper['use_ptr'],
            use_auto_trunc=model_hyper['use_auto_trunc'],
            use_font_size=model_hyper['use_font_size'],
            use_wh=model_hyper['use_wh'],
        )

        model_path = os.path.join(workdir, "best_model.pkl")
        logging.info(f"Loading model: {model_path}")
        if not os.path.isfile(model_path):
            raise ValueError(f"File not found: {model_path}")
        self.model = torch.load(model_path, map_location="cpu")
        self.model.target_vocab = feature_extractor.target_vocab
        self.model.label_vocab = feature_extractor.label_vocab

        self.tokenizer = feature_extractor.tokenizer
        self.feature_extractor = feature_extractor

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()

    def predict_cls(self, test_ds):
        cls_preds = []
        with torch.no_grad():
            for test_ins in test_ds:
                model_inp = dict()
                for k in self.model.inp_args:
                    model_inp[k] = torch.tensor(test_ins[k]).unsqueeze(0)
                cls_pred = self.model.predict_cls(**model_inp)
                cls_preds.append({
                    "doc_id": test_ins.get("doc_id"),
                    "action_id": test_ins.get("action_id"),
                    "action": cls_pred['action'],
                    "label": cls_pred['label'],
                    "stack_ptr": cls_pred['stack_ptr'],
                })
        return cls_preds

    def predict(self, data):
        parser_preds = []
        with torch.no_grad():
            for item in data:
                arcs = []
                try:
                    state = ParserState(bboxes=item['bboxes'], images=item.get("images"), model=self.model,
                                        stack_win_size=self.model.feature_extractor.stack_win_size,
                                        buffer_win_size=self.model.feature_extractor.buffer_win_size)
                    state.predict()
                    _, arcs = state.out_root.to_list()
                except ValueError as e:
                    logging.error(f"{item['doc_id']}{e}")

                parser_preds.append({
                    "doc_id": item.get("doc_id"),
                    "arcs": label_siblings(arcs)
                })
        return parser_preds


def label_siblings(_arcs):
    parents = { c: (p, l) for p, c, l in _arcs}

    _new_arcs = []
    for p, c, l in _arcs:
        if l != "sibling":
            _new_arcs.append((p, c, l))
            continue

        _p, _l = parents[c]
        path = set([c, _p])
        while _l == "sibling":
            if _p not in parents:
                break
            _p, _l = parents[_p]
            if _p in path:  # Circle
                break
            path.add(_p)
        _new_arcs.append((p, c, "sibling-" + _l))

    return _new_arcs
    

def debug_predict_cls(model_entry, workdir, ds, output_name_name):
    cls_preds = model_entry.predict_cls(ds)
    save_json(os.path.join(workdir, f"{output_name_name}-cls_preds.json"), cls_preds)
    cls_metrics = eval_cls_pred(y_true=[x['raw_target'] for x in ds], y_pred=[x['action'] for x in cls_preds],
                                labels=list(model_entry.model.target_vocab.word2idx.keys()))
    save_json(os.path.join(workdir, f"{output_name_name}-cls_metrics.json"), cls_metrics)
    return cls_preds, cls_metrics


def e2e_predict(model_entry, inp_data, workdir, output_name):
    start = time.time()
    parser_preds = model_entry.predict(inp_data)
    speed_metrics = eval_speed(time.time() - start, inp_data)

    save_json(os.path.join(workdir, f"{output_name}-parser_preds.json"), parser_preds)
    parser_metrics = eval_parser_pred(arcs_true=[x['arcs'] for x in inp_data],
                                      arcs_pred=[x['arcs'] for x in parser_preds])
    parser_metrics.update(speed_metrics)

    logging.info(f"{json.dumps(parser_metrics, indent=2)}")
    save_json(os.path.join(workdir, f"{output_name}-parser_metrics.json"), parser_metrics)
    return parser_preds, parser_metrics


def main():
    parser = argparse.ArgumentParser()
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
    main()
