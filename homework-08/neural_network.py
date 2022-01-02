import enum
import numpy as np
from enum import Enum


class NeuralNetwork:
    def __init__(self, layers_sizes, learning_rate=10):
        np.random.seed(1)
        self._layers = np.array(layers_sizes)
        self._learning_rate = learning_rate
        # init weights and layers
        self._weights = []
        self._biases = []
        for i in range(len(self._layers) - 1):
            self._weights.append(
                np.random.uniform(
                    -0.05, 0.05, size=(self._layers[i], self._layers[i + 1])
                )
            )
            self._biases.append(
                np.random.uniform(-0.05, 0.05, size=self._layers[i + 1])
            )

    def _activate(self, x):
        activated = np.zeros_like(x)
        for i, _ in enumerate(x):
            activated[i] = 1 / (1 + np.exp(-x[i]))
        return activated

    def _output_layer_errors(self, o, y):
        return np.dot(np.dot(o, 1 - o), y - o)

    def _layer_errors(self, o, delta, weights):
        errors = np.matmul(delta, weights.T)
        return np.dot(np.dot(o, 1 - o), errors)

    def _update_weights(self, idx, o, delta):
        self._weights[idx] += self._learning_rate * np.matmul(
            np.reshape(o, (-1, 1)), np.reshape(delta, (1, -1))
        )
        self._biases[idx] += self._learning_rate * delta

    def _forward_prop(self, x):
        x = np.array(x)
        outputs = [x]
        for i in range(len(self._layers) - 1):
            output = np.matmul(outputs[-1], self._weights[i]) + self._biases[i]
            outputs.append(self._activate(output))
        return outputs

    def predict(self, x):
        return self._forward_prop(x)[-1]

    def train(self, input_data, output_data, epochs=500):
        for epoch in range(epochs):
            cost = 0
            for x, y in zip(input_data, output_data):
                outputs = self._forward_prop(x)
                # calc cost
                cost += np.dot(outputs[-1] - y, outputs[-1] - y)
                # backpropagation
                last_errors = self._output_layer_errors(outputs[-1], y)
                for i in range(len(self._layers) - 2, -1, -1):
                    curr_errors = self._layer_errors(
                        outputs[i], last_errors, self._weights[i]
                    )
                    self._update_weights(i, outputs[i], last_errors)
                    last_errors = curr_errors
            cost /= 2 * len(input_data)
            print("Epoch: ", epoch, "Cost: ", cost)

        print(self._weights)
        print(self._biases)

    def test(self, input_data, output_data):
        correct = 0
        for x, y in zip(input_data, output_data):
            a = self.predict(x)
            print(x, y, a)
            guess = np.round(a[0])
            if guess == y[0]:
                correct += 1
        print("Accuracy: ", correct / len(input_data))


class BooleanOperator(Enum):
    AND = 0
    OR = 1
    XOR = 2


def generate_boolean_data(op):
    input_data, output_data = [], []
    for x in range(2):
        for y in range(2):
            input_data.append([x, y])
            if op == BooleanOperator.AND:
                res = x and y
            elif op == BooleanOperator.OR:
                res = x or y
            else:
                res = x ^ y
            output_data.append([res])
    return input_data, output_data


if __name__ == "__main__":
    nn = NeuralNetwork([2, 1, 1])
    training_data = generate_boolean_data(BooleanOperator.XOR)
    nn.train(training_data[0], training_data[1])
    test_data = generate_boolean_data(BooleanOperator.XOR)
    nn.test(test_data[0], test_data[1])
