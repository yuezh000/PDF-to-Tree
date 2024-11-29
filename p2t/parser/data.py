import os
import numpy as np
from tqdm import tqdm
from fastNLP import DataSet, Instance
from fastNLP.io import Loader
from utils import load_jsonl
import logging
from parser.state import get_train_data
from fastNLP.transformers.torch import BertTokenizer, RobertaTokenizer
from fastNLP import Vocabulary
from transformers import LayoutLMTokenizer, LayoutLMv2Tokenizer, LayoutLMv3Tokenizer, LayoutLMv3FeatureExtractor
from parser.tree import NODE_LABELS
from parser.state import DEFAULT_STACK_WINDOW_SIZE, DEFAULT_BUF_WINDOW_SIZE
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_MAX_INP_LEN = 500
SIZE_NORM = 1000
IMAGE_MODE = "RGB"


def load_doc_images(image_dir, doc_ids):
    doc_images = dict()
    for doc_id in tqdm(doc_ids):
        doc_image_dir = os.path.join(image_dir, doc_id[:2], doc_id[2:4], doc_id, "imgs")
        images = dict()
        logging.info(f"Loading images from: {doc_image_dir}")
        for file_name in os.listdir(doc_image_dir):
            if not file_name.endswith(".png"):
                continue
            page_no = file_name[:-4].split("_")[-1]
            image = Image.open(os.path.join(doc_image_dir, file_name)).convert(IMAGE_MODE).resize((225,225))
            image = np.array(image)
            image = np.transpose(image, (2, 0, 1))  # h, w, c => c, h, w
            images[page_no] = image
        doc_images[doc_id] = images
    return doc_images


def process_data(data_bundle, image_dir=None, model_name_or_path='bert-base-uncased',
                 node_labels=NODE_LABELS, max_inp_len=DEFAULT_MAX_INP_LEN, num_proc=4,
                 use_rel_pos=False, use_stack_label=False, stack_win_size=DEFAULT_STACK_WINDOW_SIZE,
                 buffer_win_size=DEFAULT_BUF_WINDOW_SIZE,
                 use_ptr=False, use_auto_trunc=False, use_font_size=False, use_wh=False):
    feature_extractor = DocLayoutFeatureExtractor(
        model_name_or_path=model_name_or_path,
        max_inp_len=max_inp_len, node_labels=node_labels,
        use_rel_pos=use_rel_pos, use_stack_label=use_stack_label,
        stack_win_size=stack_win_size, buffer_win_size=buffer_win_size,
        use_ptr=use_ptr, use_auto_trunc=use_auto_trunc,
        use_font_size=use_font_size, use_wh=use_wh
    )

    doc_images = None
    if feature_extractor.enc_type in {'layoutlmv2', 'layoutlmv3'}:
        assert image_dir is not None, f"image_dir should not be None"
        doc_ids = set()
        for ds_name in ("train", "dev", "test"):
            ds = data_bundle.get_dataset(ds_name)
            for item in ds:
                doc_ids.add(item['raw']['doc_id'])

        doc_images = load_doc_images(image_dir, doc_ids)
    
    def extract_features_and_target(raw):
        images = doc_images[raw['doc_id']] if doc_images is not None else None
        feat_and_target = feature_extractor.extract_features_and_target(raw['nodes'], raw['conf'], raw['gt_action'],
                                                                        images=images)
        feat_and_target['doc_id'] = raw['doc_id']
        feat_and_target['action_id'] = raw['action_id']

        return feat_and_target

    data_bundle.apply_field_more(extract_features_and_target, field_name='raw', num_proc=num_proc, progress_bar="rich")

    data_bundle.delete_field("raw")
    data_bundle.drop(lambda x: x['input_ids'] is None)
    data_bundle.get_dataset('train').drop(lambda x: x['ptr_target'] > stack_win_size)

    def get_inp_len(input_ids):
        return len(input_ids)

    data_bundle.apply_field(get_inp_len, field_name='input_ids', new_field_name='input_len', num_proc=num_proc)

    feature_extractor.target_vocab.index_dataset(data_bundle.datasets.values(), field_name='raw_target',
                                                 new_field_name='target')

    test_images = None
    if doc_images is not None:
        test_images = dict()
        test_doc_ids = set(data_bundle.get_dataset("test")['doc_id'])
        for doc_id in test_doc_ids:
            images = doc_images[doc_id]
            test_images[doc_id] = images

    return data_bundle, feature_extractor.tokenizer, test_images


def get_tokenizer(model_name_or_path):
    model_name = model_name_or_path.rstrip("/").split("/")[-1]
    if model_name.startswith("bert-"):
        return BertTokenizer.from_pretrained(model_name_or_path), "bert"
    elif model_name.startswith("roberta-"):
        return RobertaTokenizer.from_pretrained(model_name_or_path), "roberta"
    elif model_name.startswith("layoutlm-"):
        return LayoutLMTokenizer.from_pretrained(model_name_or_path), "layoutlm"
    elif model_name.startswith("layoutlmv2-"):
        return LayoutLMv2Tokenizer.from_pretrained(model_name_or_path), "layoutlmv2"
    elif model_name.startswith("layoutlmv3-"):
        return LayoutLMv3Tokenizer.from_pretrained(model_name_or_path), "layoutlmv3"
    else:
        raise ValueError(f"Unknown model_name_or_path: {model_name_or_path}")


class DocLayoutFeatureExtractor(object):

    def __init__(self, model_name_or_path, node_labels, max_inp_len=DEFAULT_MAX_INP_LEN,
                 stack_win_size=DEFAULT_STACK_WINDOW_SIZE, buffer_win_size=DEFAULT_BUF_WINDOW_SIZE,
                 use_rel_pos=True, use_stack_label=True, use_ptr=True, use_auto_trunc=True,
                 use_font_size=True, use_wh=True):
        self.max_inp_len = max_inp_len
        self.stack_win_size = stack_win_size
        self.buffer_win_size = buffer_win_size
        self.use_rel_pos = use_rel_pos
        self.use_stack_label = use_stack_label
        self.use_ptr = use_ptr
        self.use_auto_trunc = use_auto_trunc
        self.use_font_size=use_font_size
        self.use_wh = use_wh

        if self.use_auto_trunc:
            self.max_slot_len = int(self.max_inp_len / (self.stack_win_size + self.buffer_win_size)) - 5

        self.tokenizer, self.enc_type = get_tokenizer(model_name_or_path)
        if self.enc_type in {"layoutlm", "layoutlmv2", "layoutlmv3"}:
            if self.use_rel_pos or self.use_font_size or self.use_wh:
                raise ValueError(f"Leave position embedding for layoutlm itself to handle")
        if self.enc_type == "layoutlmv3":
            self.img_feat = LayoutLMv3FeatureExtractor(apply_ocr=False)

        self.tokenizer.add_tokens([f"[stack#{i}]" for i in range(self.stack_win_size)])
        self.tokenizer.add_tokens([f"[buffer#{i}]" for i in range(self.buffer_win_size)])
        if use_stack_label:
            self.tokenizer.add_tokens([f"[{label.lower()}]" for label in node_labels])

        # use_ptr = true,  use different vocab for label and final target
        # use_ptr = false, concatenate label nad stack_ptr as label
        self.target_vocab = Vocabulary(padding=None, unknown=None)
        if self.use_ptr:
            self.label_vocab = Vocabulary(padding=None, unknown=None)
        else:
            self.label_vocab = self.target_vocab

        for label in node_labels:
            if self.use_ptr:
                self.label_vocab.add_word(f"{label}")
            for i in range(self.stack_win_size):
                self.target_vocab.add_word(f"{label}#{i}")
                
        self.target_vocab.add_word(f"shift#-1")
        if self.use_ptr:
            self.label_vocab.add_word("shift")


    def _get_pos(self, node):
        if node is None or "x0" not in node.data:
            return [0, 0, 0, 0]
        pos = [
            round(node.x0 / node.page_width * SIZE_NORM),
            round(node.y0 / node.page_height * SIZE_NORM),
            round(node.x1 / node.page_width * SIZE_NORM),
            round(node.y1 / node.page_height * SIZE_NORM),
        ]
        assert pos[0] <= pos[2] and pos[1] <= pos[3], (pos, (node.x0, node.y0, node.x1, node.y1))
        for x in pos:
            assert 0 <= x <= SIZE_NORM, pos
        return pos

    def _align_cross_page_pos(self, node, page_nums):
        index = page_nums.index(node.page_no)
        pos = self._get_pos(node)
        offset = SIZE_NORM * index
        total = SIZE_NORM * len(page_nums)
        pos[1] = round((pos[1] + offset) / total * SIZE_NORM)
        pos[3] = round((pos[3] + offset) / total * SIZE_NORM)
        return pos

    def _get_image(self, page_nums, images):
        # Return image for page(s)
        key = "-".join([str(x) for x in page_nums])
        if key not in images:
            dst = Image.new(IMAGE_MODE, (225, 225 * len(page_nums)))
            for i, page_no in enumerate(page_nums):
                image = images[f"{page_no}"]
                image = np.transpose(image, (1, 2, 0))  # c, h, w => h, w, c
                image = Image.fromarray(image)
                dst.paste(image, (0, 225 * i))

            dst = dst.resize((225, 225))
            dst = np.array(dst)
            dst = np.transpose(dst, (2, 0, 1))  # h, w, c => c, h, w
            images[key] = dst

        if self.enc_type in {"layoutlmv3"}:
            normed_key = f"{key}-norm"
            if normed_key not in images:
                normed = self.img_feat(images[key])['pixel_values'][0]
                normed = np.array(normed)
                images[normed_key] = normed
            return images[key], images[normed_key]
        else:
            return images[key], None

    def extract_features(self, nodes, conf, images=None):
        position_pad = self._get_pos(None)
        if self.use_rel_pos:
            position_pad.extend(self._get_pos(None))
        if self.use_font_size:
            position_pad.append(0)
        if self.use_wh:
            position_pad.extend([0, 0])
        raw_words, positions, sel_indices = [], [], [0] * self.stack_win_size

        page_nums = set()
        # TODO: Skip >= 3 pages
        for item in conf['stack'][::-1] + conf['buffer']:
            for idx in item['sibling_indices']:
                node = nodes[idx]
                page_nums.add(node.page_no)
        page_nums = list(page_nums)
        assert len(page_nums) <= 2, f"TOO_MANY_PAGES: {page_nums}"

        assert conf['buffer'][0]['ptr'] == 0
        buf_top = nodes[conf['buffer'][0]['idx']]
        buf_top_pos = self._align_cross_page_pos(buf_top, page_nums)
        if self.use_font_size:
            buf_top_pos.append(int(buf_top.font_size if buf_top.font_size > 0 else 0))
        if self.use_wh:
            buf_top_pos.append(buf_top_pos[2] - buf_top_pos[0])
            buf_top_pos.append(buf_top_pos[3] - buf_top_pos[1])

        for item in conf['stack'][::-1] + conf['buffer']:
            if item['type'].lower() == "buffer" and item['ptr'] == 0:
                raw_words.append(self.tokenizer.sep_token)
                positions.append(position_pad)

            raw_words.append(f"[{item['type'].lower()}#{item['ptr']}]")
            positions.append(position_pad)

            if item['type'].lower() == "stack":
                sel_indices[item['ptr']] = len(raw_words) - 1

            if self.use_stack_label and 'label' in item:
                if item['label'] is not None:
                    raw_words.append(f"[{item['label'].lower()}]")
                    positions.append(position_pad)

            raw_words_i, positions_i = [], []
            for idx in item['sibling_indices']:
                node = nodes[idx]
                tokens = self.tokenizer.tokenize(node.text)
                pos = self._align_cross_page_pos(node, page_nums)
                if self.use_font_size:
                    pos.append(int(node.font_size if node.font_size > 0 else 0))
                if self.use_wh:
                    pos.append(pos[2] - pos[0])
                    pos.append(pos[3] - pos[1])
                if self.use_rel_pos:
                    rel_pos = [(SIZE_NORM + x1 - x2) for x1, x2 in zip(pos[:4], buf_top_pos[:4])]
                    for x in pos:
                        assert 0 <= x < 2 * SIZE_NORM, f"POS: {pos}"
                    pos.extend(rel_pos)

                assert len(pos) == len(position_pad), (pos, position_pad)
                raw_words_i.extend(tokens)
                positions_i.extend([pos for _ in tokens])

            if self.use_auto_trunc and len(raw_words_i) > self.max_slot_len:
                logging.info(f"Truncate: {item['type']}#{item['ptr']}: {len(raw_words_i)}")
                if item['type'].lower() == "buffer":
                    raw_words_i, positions_i = raw_words_i[:self.max_slot_len], positions_i[:self.max_slot_len]
                else:
                    raw_words_i, positions_i = raw_words_i[-self.max_slot_len:], positions_i[-self.max_slot_len:]

            raw_words.extend(raw_words_i)
            positions.extend(positions_i)

        assert len(raw_words) <= self.max_inp_len, f"TOO_LONG: len(raw_words) = {len(raw_words)}"
        for sel_idx in sel_indices:
            assert raw_words[sel_idx].startswith("[stack#"), f"SEL_IDX_ERR: {raw_words[sel_idx]}"

        input_ids = self.tokenizer.convert_tokens_to_ids(raw_words)
        assert input_ids[0] != self.tokenizer.unk_token_id, f"UNK: {raw_words[0]}"
        assert len(input_ids) == len(raw_words), \
            f"TOKENIZER_ERR: len(input_ids) = {len(input_ids)}, len(raw_words)={len(raw_words)}"

        input_ids.insert(0, self.tokenizer.cls_token_id)
        positions.insert(0, position_pad)

        image, normed = None, None
        if images is not None:
            image, normed = self._get_image(page_nums, images)

        return {
            "raw_words": raw_words,  # ... stack#1, stack #0
            "input_ids": input_ids,
            "positions": positions,
            "attention_mask": [1] * len(input_ids),
            "sel_indices": sel_indices,  # stack#0, stack#1 ...
            "image": image,
            "pixel_values": normed,
            "page_nums": page_nums
        }

    def extract_features_and_target(self, nodes, conf, target, images=None):
        label, stack_ptr = target
        target = f"{label}#{stack_ptr}"

        try:
            result = self.extract_features(nodes, conf, images)
        except (ValueError, AssertionError) as e:
            # logging.warning(f"  Skip item for: {e}")
            result = {
                "raw_words": None,
                "input_ids": None,
                "positions": None,
                "attention_mask": None,
                "sel_indices": None,
                "image": None,
                "pixel_values": None,
                "page_nums": None
            }

        result['raw_target'] = target
        result['ptr_target'] = stack_ptr
        if self.use_ptr:
            result['raw_label_target'] = label
            label_target = [-1] * self.stack_win_size
            if label == "shift":
                # Set the labels of all stack element to shift
                for i in range(len(conf['stack'])):
                    label_target[i] = self.label_vocab.to_index(label)
            else:
                label_target[stack_ptr] = self.label_vocab.to_index(label)
            result['label_target'] = label_target
        return result


class DocLayoutLoader(Loader):

    def __init__(self, stack_win_size=DEFAULT_STACK_WINDOW_SIZE, buffer_win_size=DEFAULT_BUF_WINDOW_SIZE):
        super(DocLayoutLoader, self).__init__()
        self.stack_win_size = stack_win_size
        self.buffer_win_size = buffer_win_size

    def _load(self, path):
        ds = DataSet()

        raw_data = load_jsonl(path)
        for raw_data_item in raw_data:
            bboxes, gt_arcs = raw_data_item['bboxes'], raw_data_item['arcs']

            gt_arcs = [(h, t, "sibling" if l.startswith("sibling-") else l) for h, t, l in gt_arcs]

            history, nodes = get_train_data(gt_arcs, bboxes, stack_win_size=self.stack_win_size,
                                            buffer_win_size=self.buffer_win_size)

            for i, (conf, gt_action) in enumerate(history):
                if gt_action[0] == "reverse":
                    continue
                ds.append(Instance(**{
                    "raw": {
                        "doc_id": raw_data_item["doc_id"].replace(".json", ""),
                        "action_id": i,
                        "nodes": nodes,
                        "conf": conf,
                        "gt_action": gt_action
                    }
                }))

        return ds


