import logging

from parser.tree import TreeNode


DEFAULT_STACK_WINDOW_SIZE = 3
DEFAULT_BUF_WINDOW_SIZE = 1


class ParserState(object):

    def __init__(self, bboxes, images=None, model=None, stack_win_size=DEFAULT_STACK_WINDOW_SIZE,
                 buffer_win_size=DEFAULT_BUF_WINDOW_SIZE):
        self.out_root = TreeNode({
            "index": 0,
            "page_no": 0,
            "font_size": 0,
            "label": "ROOT",
            "text": "ROOT"
        })
        self.stack = [self.out_root]
        self.buffer = list([TreeNode(bbox) for bbox in bboxes])
        self.nodes = dict([(node.index, node) for node in self.stack + self.buffer])
        self.images = images
        self.model = model
        self.history = []
        self.stack_win_size = stack_win_size
        self.buffer_win_size = buffer_win_size
        self.label_mask = {}

    def get_conf(self):
        # Create snapshot of parser state, using only node indices
        return {
            "stack": list([{
                "ptr": i,
                "type": "stack",
                "idx": node.index,
                "label": node.label,
                "sibling_indices": node.sibling_indices
            } for i, node in enumerate(self.stack[:min(self.stack_win_size, len(self.stack))])]),
            "buffer": list([{
                "ptr": i,
                "type": "buffer",
                "idx": node.index,
                "sibling_indices": node.sibling_indices
            } for i, node in enumerate(self.buffer[:min(self.buffer_win_size, len(self.buffer))])]),
        }

    def apply(self, label, stack_ptr):
        self.history.append((self.get_conf(), (label, stack_ptr)))
        if label == "shift":
            self.shift()
        elif label == "reverse":
            self.reverse()
        else:
            self.reduce(label, stack_ptr)

    def shift(self):
        self.stack.insert(0, self.buffer.pop(0))

    def reverse(self):
        self.buffer = list(self.stack)
        self.stack = list([self.buffer.pop(0)])

    def reduce(self, label, stack_ptr):
        if stack_ptr >= len(self.stack) or stack_ptr < 0:
            logging.warning(f"Incorrect stack_ptr: {stack_ptr}")
            self.shift()
            return
        src = self.stack[stack_ptr]
        dst = self.buffer[0]
        if label == "sibling":
            src.append_sibling(dst)
        else:
            src.append_child(dst)
            self.stack.insert(0, dst)
        dst.label = label
        self.buffer.pop(0)

    def predict_one_pass(self):
        while len(self.buffer) > 0:
            conf = self.get_conf()
            try:
                label, stack_ptr = self.model.predict_action(nodes=self.nodes, conf=conf, images=self.images)
                self.apply(label, stack_ptr)
            except (ValueError, AssertionError):
                self.apply("shift", -1)

    def connect_orphans(self):
        assert len(self.buffer) == 0
        ptr = self.out_root
        
        self.stack.remove(self.out_root)
        # Remove all descendants of root from stack
        while ptr.child is not None:
            self.stack.remove(ptr.child)
            ptr = ptr.child

        self.stack = self.stack[::-1]

        # Find all orphan nodes and append to main thread as "unknown" nodes
        while len(self.stack) > 0:
            orphan = None
            for node in self.stack:
                if node.parent is None:
                    orphan = node
                    self.stack.remove(orphan)
                    break
            orphan.label = "unknown"
            ptr.append_child(orphan)

            # Remove all descendants of top from stack
            while orphan.child is not None:
                self.stack.remove(orphan.child)
                orphan = orphan.child
            ptr = orphan

    def predict(self):
        # 第一阶段, 正向，
        self.predict_one_pass()

        self.apply("reverse", -1)

        # 第二阶段，逆向
        self.predict_one_pass()

        # 将stack中剩余的node挂在文档末尾
        self.connect_orphans()


def get_train_data(gt_arcs, bboxes, stack_win_size=DEFAULT_STACK_WINDOW_SIZE,
                   buffer_win_size=DEFAULT_BUF_WINDOW_SIZE):
    oracle = OracleClassifier(gt_arcs)
    state = ParserState(bboxes, model=oracle, stack_win_size=stack_win_size, buffer_win_size=buffer_win_size)
    state.predict()

    return state.history, state.nodes


class OracleClassifier(object):

    def __init__(self, gt_arcs, stack_win_size=DEFAULT_STACK_WINDOW_SIZE):
        self.gt_arcs = dict([((src, dst), label)for (src, dst, label) in gt_arcs])
        self.stack_win_size = stack_win_size

    def predict_action(self, nodes, conf, images):
        dst_idx = conf['buffer'][0]['idx']
        stack_len = len(conf['stack'])
        for stack_ptr in range(stack_len):
            src_idx = conf['stack'][stack_ptr]['idx']
            elder_idx = conf['stack'][stack_ptr]['sibling_indices'][-1]
            label = None
            if self.gt_arcs.get((elder_idx, dst_idx)) == "sibling":
                label = "sibling"
            elif self.gt_arcs.get((src_idx, dst_idx)) is not None:
                label = self.gt_arcs.get((src_idx, dst_idx))
            if label is not None:
                if stack_ptr > self.stack_win_size:
                    print(f"WARNING: stack_win_size={self.stack_win_size} is too small, required: {stack_ptr}")
                    break
                else:
                    return label, stack_ptr

        return "shift", -1

