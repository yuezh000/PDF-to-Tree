from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict

def eval_cls_pred(y_true, y_pred, labels):
    return {
        'f1': f1_score(y_true=y_true, y_pred=y_pred, average="weighted"),
        'precision': precision_score(y_true=y_true, y_pred=y_pred, average="weighted"),
        'recall': recall_score(y_true=y_true, y_pred=y_pred, average="weighted"),
        'f1#by-class': f1_score(y_true=y_true, y_pred=y_pred, labels=labels, average=None).tolist(),
        'p#by-class': precision_score(y_true=y_true, y_pred=y_pred, labels=labels, average=None).tolist(),
        'r#by-class': recall_score(y_true=y_true, y_pred=y_pred, labels=labels, average=None).tolist(),
    }


def arcs_to_labels(max_index, arcs):
    labels = ['unknown'] * (max_index + 1)
    for h, t, l in arcs:
        assert t < len(labels), (t, len(labels), max_index)
        labels[t] = l
    return labels[1:]  # Skip root node


def filter_sibling(arcs):
    new_arcs = []
    for p, c, l in arcs:
        if l.startswith("sibling-"):
            l = l[len('sibling-'):]
        new_arcs.append((p, c, l))
    return new_arcs


def eval_parser_pred(arcs_true, arcs_pred):
    arcs_true = [filter_sibling(ins) for ins in arcs_true]
    arcs_pred = [filter_sibling(ins) for ins in arcs_pred]

    # Attachment Score
    uas_counter = defaultdict(int)
    las_counter = defaultdict(int)
    label_counter = defaultdict(int)
    for _arcs_true, _arcs_pred in zip(arcs_true, arcs_pred):
        arcs_true_map = dict(((p, c), l) for p, c, l in _arcs_true)
        arcs_pred_map = dict(((p, c), l) for p, c, l in _arcs_pred)
        for p, c, l in _arcs_true:
            label_counter[l] += 1
            if (p, c) in arcs_pred_map:
                uas_counter[l] += 1
                if l == arcs_pred_map[(p, c)]:
                    las_counter[l] += 1

    uas_total = sum(uas_counter.values())
    las_total = sum(las_counter.values())
    total = sum(label_counter.values())

    uas = uas_total / total
    las = las_total / total

    gt_labels, pred_labels = [], []
    for gt_arcs, pred_arcs in zip(arcs_true, arcs_pred):
        max_index = max([t for _, t, _ in gt_arcs])
        gt_labels.extend(arcs_to_labels(max_index, gt_arcs))
        pred_labels.extend(arcs_to_labels(max_index, pred_arcs))

    f1 = f1_score(gt_labels, pred_labels, average="weighted")
    p = precision_score(gt_labels, pred_labels, average="weighted")
    r = recall_score(gt_labels, pred_labels, average="weighted")

    metrics = {
        "uas": uas,
        "las": las,
        "f1": f1,
        "p": p,
        "r": r,
    }
    return metrics


def eval_speed(total_time, inp_data):
    total_page = 0
    for item in inp_data:
        total_page += len(set([b['page_no'] for b in item['bboxes']]))
    time_per_page = round(total_time / total_page, 2)
    pages_per_sec = round(total_page / total_time, 2)
    total_time = round(total_time, 2)

    return {
        "total_time": total_time,
        "total_page": total_page,
        "time_per_page": time_per_page,
        "pages_per_sec": pages_per_sec,
    }

