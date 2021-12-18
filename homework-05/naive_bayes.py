import pandas as pd
import numpy as np
import math


class NaiveBayesClassifier:
    NUM_CLASSES = 2
    CROSS_VAL = 10

    def encode_dataset(self, dataset):
        dataset.replace("republican", 0, True)
        dataset.replace("democrat", 1, True)
        dataset.replace("y", 2, True)
        dataset.replace("n", 0, True)
        dataset.replace("?", 1, True)
        return dataset

    def init(self):
        self.dataset = pd.read_csv(
            "./house-votes-84.data",
            header=None,
        )

        self.dataset = self.encode_dataset(self.dataset)
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

    def train(self, train_dataset):
        num_cols = train_dataset.shape[1]
        probs = np.ndarray((num_cols - 1, 2, 3))

        for i in range(1, num_cols):
            for cls in range(self.NUM_CLASSES):
                df = train_dataset[train_dataset[0] == cls]
                freq = df.copy().groupby(by=i).count() / len(df)
                for j in range(3):
                    probs[i - 1][cls][j] = freq[0][j]

        class_freq = [0] * self.NUM_CLASSES
        for cls in range(self.NUM_CLASSES):
            class_freq[cls] = len(train_dataset[train_dataset[0] == cls])

        return probs, class_freq

    def evaluate(self, probs, class_freq, val_dataset):
        correct, wrong = 0, 0
        for _, row in val_dataset.iterrows():
            max_prob, max_cls = -1, -1
            for cls in range(self.NUM_CLASSES):
                prob = class_freq[cls]
                for j in range(1, len(row)):
                    prob *= probs[j - 1][cls][row[j]]
                if prob > max_prob:
                    max_prob, max_cls = prob, cls

            if max_cls == row[0]:
                correct += 1
            else:
                wrong += 1
        return correct / (correct + wrong)

    def run(self):
        val_size = math.floor(len(self.dataset) / self.CROSS_VAL)
        sum_accuracy = 0
        for i in range(self.CROSS_VAL):
            start, end = val_size * i, min((i + 1) * val_size, len(self.dataset))
            train_dataset = pd.concat(
                [self.dataset[:start], self.dataset[end:]], ignore_index=True
            )
            val_dataset = self.dataset[start:end]
            probs, class_freq = self.train(train_dataset)
            accuracy = self.evaluate(probs, class_freq, val_dataset)
            print("Fold: ", i, "Accuracy: ", accuracy)
            sum_accuracy += accuracy

        print("Average accuracy: ", sum_accuracy / self.CROSS_VAL)


if __name__ == "__main__":
    nbc = NaiveBayesClassifier()
    nbc.init()
    nbc.run()
