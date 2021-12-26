import pandas as pd
import numpy as np
import math
from pandas.io import json
from scipy.sparse import data
from sklearn.preprocessing import OrdinalEncoder


class Node:
    def __init__(self, cls, attr, children, id):
        self.cls = cls
        self.attr = attr
        self.children = children
        self.id = id


class ID3Classifier:
    CROSS_VAL = 10
    MIN_SIZE_SET = 3

    def encode_dataset(self, dataset):

        self.encoder = OrdinalEncoder()
        return self.encoder.fit_transform(dataset)

    def init(self):
        self.dataset = pd.read_csv(
            "./breast-cancer.data",
            header=None,
        )

        self.dataset = self.encode_dataset(self.dataset)
        self.dataset = pd.DataFrame(self.dataset)

        np.random.seed(3)
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

    def entropy(self, xs):
        xs = xs / np.sum(xs)
        return -np.sum([x * np.log(x) if x else 0 for x in xs])

    def target_entropy(self, x):
        _, counts = np.unique(x[0], return_counts=True)
        return self.entropy(counts)

    def build_tree(self, train_dataset, excluded_cols=[]):
        target_entropy = self.target_entropy(train_dataset)
        if (
            target_entropy == 0
            or len(train_dataset.columns) == 1 + len(excluded_cols)
            or len(train_dataset) <= self.MIN_SIZE_SET
        ):
            self.node_cnt += 1
            return Node(round(np.mean(train_dataset[0])), None, {}, self.node_cnt)

        best_gain, best_col = -1, 0

        for column in train_dataset.columns[1:]:
            if column in excluded_cols:
                continue
            grouped = train_dataset.groupby(column).apply(
                lambda x: x.count() * self.target_entropy(x)
            )[0]
            gain = target_entropy - np.sum(grouped) / len(train_dataset)
            if gain > best_gain:
                best_gain, best_col = gain, column

        next_excluded = excluded_cols + [best_col]

        children = {}
        for attr_cls, next_dataset in train_dataset.groupby(best_col):
            children[attr_cls] = self.build_tree(next_dataset, next_excluded)

        self.node_cnt += 1
        return Node(None, best_col, children, self.node_cnt)

    def print_tree(self, root_node):
        result = ""
        result += f"{root_node.id}[label={root_node.attr or root_node.cls}];\n"
        for child_key, child_node in root_node.children.items():
            result += f'{root_node.id} -> {child_node.id} [label="{child_key}"];\n'
            result += self.print_tree(child_node)
        return result

    def train(self, train_dataset):
        self.node_cnt = 0
        return self.build_tree(train_dataset)

    def predict(self, row, current_node):
        if current_node.cls is not None:
            return current_node.cls
        if row[current_node.attr] not in current_node.children:
            for _, v in current_node.children.items():
                return self.predict(row, v)
        return self.predict(row, current_node.children[row[current_node.attr]])

    def evaluate(self, root_node, val_dataset):
        correct, wrong = [0, 0], [0, 0]
        for _, row in val_dataset.iterrows():
            predicted = self.predict(row, root_node)

            if predicted == row[0]:
                correct[int(row[0])] += 1
            else:
                wrong[int(row[0])] += 1
        print(np.sum(val_dataset[0] == 0) / len(val_dataset))
        return (
            correct[0] / (correct[0] + wrong[0]) / 2
            + correct[1] / (correct[1] + wrong[1]) / 2
        ), (correct[0] + correct[1]) / len(val_dataset)

    def run(self):
        val_size = math.floor(len(self.dataset) / self.CROSS_VAL)
        sum_accuracy = 0
        sum_bal_acc = 0
        for i in range(self.CROSS_VAL):
            start, end = val_size * i, min((i + 1) * val_size, len(self.dataset))
            train_dataset = pd.concat(
                [self.dataset[:start], self.dataset[end:]], ignore_index=True
            )
            val_dataset = self.dataset[start:end]
            root = self.train(train_dataset)
            balanced_acc, accuracy = self.evaluate(root, val_dataset)
            print(
                "Fold: ", i, "Accuracy: ", accuracy, "Balanced accuracy: ", balanced_acc
            )
            sum_accuracy += accuracy
            sum_bal_acc += balanced_acc

            if i == 0:
                with open("./decision-tree.dot", "w") as f:
                    file_cont = f"digraph{{\n{self.print_tree(root)}}}"
                    f.write(file_cont)

        print("Average accuracy: ", sum_accuracy / self.CROSS_VAL)
        print("Balanced accuracy: ", sum_bal_acc / self.CROSS_VAL)


if __name__ == "__main__":
    id3 = ID3Classifier()
    id3.init()
    id3.run()
