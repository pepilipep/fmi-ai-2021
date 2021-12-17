import pandas as pd
import numpy as np

# names=[
#     "class",
#     "handicapped-infants",
#     "water-project-cost-sharing",
#     "adoption-of-the-budget-resolution",
#     "physician-fee-freeze",
#     "el-salvador-aid",
#     "religious-groups-in-schools",
#     "anti-satellite-test-ban",
#     "aid-to-nicaraguan-contras",
#     "mx-missile",
#     "immigration",
#     "synfuels-corporation-cutback",
#     "education-spending",
#     "superfund-right-to-sue",
#     "crime",
#     "duty-free-exports",
#     "export-administration-act-south-africa",
# ],


class NaiveBayesClassifier:
    NUM_CLASSES = 2

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

    def train(self):
        num_cols = self.dataset.shape[1]
        probs = np.ndarray((num_cols - 1, 2, 3))

        for i in range(1, num_cols):
            for cls in range(self.NUM_CLASSES):
                df = self.dataset[self.dataset[0] == cls]
                freq = df.copy().groupby(by=i).count() / len(df)
                print(freq)
                for j in range(3):
                    probs[i - 1][cls][j] = freq[0][j]


if __name__ == "__main__":
    nbc = NaiveBayesClassifier()
    nbc.init()
    nbc.train()
