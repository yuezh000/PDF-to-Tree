from collections import defaultdict
from copy import deepcopy


class TreeNode(object):

    @staticmethod
    def from_list(bboxes, arcs):
        tree_nodes = list([TreeNode(node) for node in bboxes])
        tree_nodes_map = dict([(tn.index, tn) for tn in tree_nodes])
        for p_idx, c_idx, label in arcs:
            parent = tree_nodes_map[p_idx]
            child = tree_nodes_map[c_idx]
            child.label = label
            if label == "sibling":
                parent.append_sibling(child)
            else:
                parent.append_child(child)
        return tree_nodes_map[0]

    def __init__(self, data):
        self.data = deepcopy(data)
        self.index = self.data.get('index')
        self.label = self.data.get('label')
        self.parent = None
        self.child = None
        self.sibling = None
        self.sibling_indices = (self.index,)
        self.last_sibling = self

    def to_list(self):
        visited = {self.index}
        nodes, arcs = [], []
        queue = [self]
        while len(queue) > 0:
            node = queue.pop(0)
            nodes.append(node.to_json())
            if node.child is not None:
                arcs.append((node.index, node.child.index, node.child.label))
                if node.child.index not in visited:
                    visited.add(node.child.index)
                    queue.append(node.child)
            if node.sibling is not None:
                arcs.append((node.index, node.sibling.index, node.sibling.label))
                if node.sibling.index not in visited:
                    visited.add(node.sibling.index)
                    queue.append(node.sibling)

        return nodes, arcs

    def append_child(self, child):
        if self.child is not None:
            raise ValueError("Too many children")
        if child.parent is not None:
            raise ValueError("Too many parents")
        ptr = self.parent
        while ptr is not None:
            if ptr == child:
                raise ValueError("Circle detected")
            ptr = ptr.parent
        self.child = child
        child.parent = self

    def append_sibling(self, sibling):
        if self.parent == sibling:
            raise ValueError(f"Sibling circle")
        if sibling.parent is not None:
            curr_parent = self.parent
            sibling.parent.append_child(self)
            curr_parent.child = None
        if sibling.child is not None:
            sibling.child.parent = None
            sibling.child = None
        sibling.label = "sibling"
        self.last_sibling.sibling = sibling
        self.sibling_indices = self.sibling_indices + sibling.sibling_indices
        if sibling.last_sibling is not None:
            self.last_sibling = sibling.last_sibling
        else:
            self.last_sibling = sibling

    def pretty_tree(self, indent=0, visited=None):
        output = f"{self.__repr__()}\n"
        visited = visited or set()
        visited.add(self)
        if self.label == "sibling":
            output = f"{' ' * indent}{output}"
        if self.sibling is not None and self.sibling not in visited:
            output += self.sibling.pretty_tree(indent, visited)
        if self.child is not None and self.child not in visited:
            output += self.child.pretty_tree(indent, visited)
        return output

    def to_json(self):
        json_data = deepcopy(self.data)
        json_data['index'] = self.index
        # json_data['label'] = self.label
        if 'type' in json_data:
            del json_data['type']
        return json_data

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError(f"Instance {self.index} doesn't have '{key}' attribute")
        return self.data[key]

    def __iter__(self):
        yield self
        if self.child is not None:
            for x in self.child:
                yield x
        if self.sibling is not None:
            for x in self.sibling:
                yield x

    def __str__(self):
        return self.pretty_tree(2)

    def __repr__(self):
        return f"({self.index}[{self.label}]{self.text})"


# NODE_LABELS = [
#   "root",
#   "header",
#   "question",
#   "other",
#   "sibling",
# ]

NODE_LABELS = [
    'paragraph',
    'other',
    'meta',
    'content',
    'title',
    'reference',
    'table_text',
    'sibling',
    'annotation',
    'page_num',
    'header',
    'equation',
    'caption',
    'footer',
    'figure_text',
    'answer',
    'question',
    'person_info',
    'section'
 ]