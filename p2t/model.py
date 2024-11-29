from collections import defaultdict
import torch
from torch import nn
from fastNLP.transformers.torch import BertModel
from fastNLP import logger
from transformers import LayoutLMModel, LayoutLMv2Model, LayoutLMv3Model, RobertaModel


def get_model(model_name_or_path, embedding_size,  alpha=0.3, max_pos=2000):
    model_name = model_name_or_path.rstrip("/").split("/")[-1]
    if model_name.startswith("bert-"):
        return PositionAwareEncoder(model_name_or_path, BertModel, embedding_size,
                                        alpha=alpha, max_pos=max_pos), "bert"
    elif model_name.startswith("roberta-"):
        return PositionAwareEncoder(model_name_or_path, RobertaModel, embedding_size,
                                        alpha=alpha, max_pos=max_pos), "roberta"
    elif model_name.startswith("layoutlm-"):
        return LayoutLMEncoder(model_name_or_path, embedding_size), "layoutlm"
    elif model_name.startswith("layoutlmv2-"):
        return LayoutLMv2Encoder(model_name_or_path, embedding_size), "layoutlmv2"
    elif model_name.startswith("layoutlmv3-"):
        return LayoutLMv3Encoder(model_name_or_path, embedding_size), "layoutlmv3"
    else:
        raise ValueError(f"Unknown model_name_or_path: {model_name_or_path}")


class BaseEncoder(nn.Module):

    def __init__(self):
        super().__init__()

    @property
    def hidden_size(self):
        return self.encoder.config.hidden_size

    @property
    def device(self):
        return self.encoder.device


class LayoutLMEncoder(BaseEncoder):

    def __init__(self, model_name_or_path, embedding_size):
        super().__init__()
        self.encoder = LayoutLMModel.from_pretrained(model_name_or_path)
        self.encoder.resize_token_embeddings(embedding_size)

    def forward(self, input_ids, positions, attention_mask):
        outputs = self.encoder(input_ids=input_ids, bbox=positions, attention_mask=attention_mask)
        return outputs.pooler_output, outputs.last_hidden_state


class LayoutLMv2Encoder(BaseEncoder):

    def __init__(self, model_name_or_path, embedding_size):
        super().__init__()
        self.encoder = LayoutLMv2Model.from_pretrained(model_name_or_path)
        self.encoder.resize_token_embeddings(embedding_size)

    def forward(self, input_ids, positions, attention_mask, image):
        outputs = self.encoder(input_ids=input_ids, bbox=positions, attention_mask=attention_mask, image=image)
        return outputs.pooler_output, outputs.last_hidden_state


class LayoutLMv3Encoder(BaseEncoder):

    def __init__(self, model_name_or_path, embedding_size):
        super().__init__()
        self.encoder = LayoutLMv3Model.from_pretrained(model_name_or_path)
        self.encoder.resize_token_embeddings(embedding_size)

        # For some reason, pooler is not implemented in LayoutLMv3Model.
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, input_ids, positions, attention_mask, pixel_values):
        outputs = self.encoder(input_ids=input_ids, bbox=positions, attention_mask=attention_mask,
                               pixel_values=pixel_values)

        first_token_tensor = outputs.last_hidden_state[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output, outputs.last_hidden_state


class PositionAwareEncoder(BaseEncoder):

    def __init__(self, model_name_or_path, model_cls, embedding_size,  alpha=0.3, max_pos=2000):
        super(PositionAwareEncoder, self).__init__()

        self.alpha = alpha
        self.max_pos = max_pos

        self.encoder = model_cls.from_pretrained(model_name_or_path)
        self.pos_embedding = nn.Embedding(max_pos, self.encoder.config.hidden_size)

        self.encoder.resize_token_embeddings(embedding_size)

    def forward(self, input_ids, positions, attention_mask):
        word_embs = self.encoder.embeddings(input_ids=input_ids)  # (B, S, H)
        pos_embs = self.pos_embedding(positions)  # (B, S, P, H)
        mean_pos_embs = torch.mean(pos_embs, dim=-2)  # (B, S, H)

        inputs_embeds = (1 - self.alpha) * word_embs + self.alpha * mean_pos_embs  # ( B, S, H)

        outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        return outputs.pooler_output, outputs.last_hidden_state


class DocLayoutParserModel(nn.Module):

    def __init__(self, model_name_or_path, feature_extractor, use_ptr=False,
                 max_pos=2000, dropout=0.3, alpha=0.3, beta=0.5):
        super(DocLayoutParserModel, self).__init__()

        self.feature_extractor = feature_extractor
        self.target_vocab = feature_extractor.target_vocab
        self.label_vocab = feature_extractor.label_vocab
        self.tokenizer = feature_extractor.tokenizer
        self.dropout = dropout
        self.alpha = alpha
        self.beta = beta

        self.encoder, self.enc_type = get_model(model_name_or_path=model_name_or_path, embedding_size=len(self.tokenizer),
                                                alpha=alpha, max_pos=max_pos)
        if self.enc_type in {"bert", "roberta", "layoutlm"}:
            self.inp_args = ['input_ids', 'positions', 'attention_mask']
        elif self.enc_type in {"layoutlmv2"}:
            self.inp_args = ['input_ids', 'positions', 'attention_mask', 'image']
        elif self.enc_type in {"layoutlmv3"}:
            self.inp_args = ['input_ids', 'positions', 'attention_mask', 'pixel_values']
        else:
            raise ValueError(f"Unknown enc_type : {self.enc_type}")

        self.tgt_args = ['target']
        if use_ptr:
            self.inp_args.extend(['sel_indices'])
            self.tgt_args.extend(['label_target', 'ptr_target'])
            self.get_loss = self.get_loss_with_ptr
            self.get_logits = self.get_logits_with_ptr
            self.predict_cls = self.predict_cls_with_ptr
            self.num_labels = len(self.label_vocab)
        else:
            self.num_labels = len(self.target_vocab)

        self.classifier = nn.Sequential(nn.Linear(self.encoder.hidden_size, self.encoder.hidden_size),
                                        nn.Dropout(dropout),
                                        nn.Linear(self.encoder.hidden_size, self.num_labels))

        self.soft_max = nn.Softmax(dim=-1)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

        logger.info(f"Model\n  encoder: {self.enc_type}, {model_name_or_path}")
        logger.info(f"  input: {self.inp_args}")
        logger.info(f"  target: {self.tgt_args}")

    @property
    def train_input_mapping(self):
        return dict([(f, f) for f in self.inp_args + self.tgt_args])

    @property
    def evaluate_input_mapping(self):
        return dict([(f, f) for f in self.inp_args])

    def get_logits(self, **inp):
        pooler_output, _ = self.encoder(**inp)
        return self.classifier(pooler_output)  # (B, L)

    def get_loss(self, target, **inp):
        logits = self.get_logits(**inp)
        return self.ce_loss(logits, target)

    def get_logits_with_ptr(self, sel_indices, **inp):
        _, last_hidden_state = self.encoder(**inp)
        sel_indices_ex = sel_indices.unsqueeze(-1).expand(-1, -1, last_hidden_state.shape[-1])
        selected_hidden_states = last_hidden_state.gather(dim=1, index=sel_indices_ex)

        return self.classifier(selected_hidden_states)  # B, len(stack), L

    def get_loss_with_ptr(self, target, label_target, ptr_target, **inp):
        # label_target: B, len(stack)
        # ptr_target: B
        label_logits = self.forward(**inp)  # B, len(stack), L
        label_loss = self.ce_loss(label_logits.view(-1, label_logits.shape[-1]), label_target.view(-1))

        assert not torch.isnan(label_loss), (label_logits, label_target)

        ptr_logits, _ = torch.max(label_logits, dim=-1)  # B, len(stack)
        ptr_loss = self.ce_loss(ptr_logits, ptr_target)

        if torch.isnan(ptr_loss):
            ptr_loss = 0

        return (1 - self.beta) * label_loss + self.beta * ptr_loss

    def forward(self, **inp):
        return self.get_logits(**inp)

    def predict_action(self, nodes, conf, images=None):
        features = self.feature_extractor.extract_features(nodes, conf, images)
        model_inp = dict()
        for field_name in self.inp_args:
            t = torch.tensor(features[field_name])
            model_inp[field_name] = t.unsqueeze(0)
        output = self.predict_cls(**model_inp)
        # print(output['action'])
        return output['label'], output['stack_ptr']

    def predict_cls(self, **model_inp):
        for k, t in model_inp.items():
            model_inp[k] = t.to(self.encoder.device)
        logits = self.forward(**model_inp)[0]

        logits = self.soft_max(logits)  # (B, L)
        index = torch.argmax(logits)
        index = index.detach().cpu().item()
        action = self.target_vocab.to_word(index)
        label, stack_ptr = action.split("#")
        stack_ptr = int(stack_ptr)

        return {
            'pred': index,
            'action':  action,
            'label': label,
            'stack_ptr': stack_ptr,
        }

    def predict_cls_with_ptr(self, **model_inp):
        for k, t in model_inp.items():
            model_inp[k] = t.to(self.encoder.device)

        label_logits = self.forward(**model_inp)[0]  # len(stack), L

        ptr_logits, label_indices = torch.max(label_logits, dim=-1)  # len(stack)

        ptr_pred = torch.argmax(ptr_logits, dim=-1)
        label_pred = label_indices[ptr_pred]

        ptr_pred = ptr_pred.detach().cpu().item()
        label_pred = label_pred.detach().cpu().item()
        label = self.label_vocab.to_word(label_pred)
        if label == "shift":
            ptr_pred = -1

        return {
            "label_pred": label_pred,
            "action": f"{label}#{ptr_pred}",
            "label": label,
            "stack_ptr": ptr_pred,
        }

    def train_step(self, **inp):
        train_inp = dict()
        for k in self.inp_args + self.tgt_args:
            train_inp[k] = inp[k]
        return {
            "loss": self.get_loss(**train_inp)
        }

    def evaluate_step(self, **eval_inp):
        model_inps = [dict() for _ in eval_inp['input_ids']]

        for k in self.inp_args:
            batch = eval_inp[k]
            for i, v in enumerate(batch):
                model_inps[i][k] = v.unsqueeze(0)

        pred = []
        for model_inp in model_inps:
            outputs = self.predict_cls(**model_inp)
            pred.append(self.target_vocab.to_index(outputs['action']))
        return {
            "pred": torch.tensor(pred),
        }


def tensor_print(t, name=""):
    assert type(t) == torch.Tensor, type(t)
    print(f"name={name}, max={torch.max(t)}, min={torch.min(t)}, mean={torch.mean(t)}")
