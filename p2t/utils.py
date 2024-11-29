import json
import pickle


def load_json(file_path):
    with open(file_path) as inp_file:
        return json.loads(inp_file.read())


def save_json(file_path, data):
    with open(file_path, 'w') as out_file:
        out_file.write(json.dumps(data, indent=2, ensure_ascii=False))


def load_jsonl(file_path):
    with open(file_path) as inp_file:
        return list([json.loads(line) for line in inp_file])


def save_jsonl(file_path, data):
    with open(file_path, 'w') as out_file:
        for item in data:
            out_file.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_pickle(obj, out_path):
    with open(out_path, 'wb') as out_file:
        pickle.dump(obj, out_file)


def load_pickle(inp_path, cls=None):
    with open(inp_path, 'rb') as inp_file:
        obj = pickle.load(inp_file)
        if cls is not None:
            assert type(obj) == cls,  f"{type(obj)} is not {cls}"
    return obj
