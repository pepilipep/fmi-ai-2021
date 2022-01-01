import sys
import math
import numpy as np
import random
import matplotlib.pyplot as plt


class KMeans:

    NUM_RESTARTS = 30

    def init(self, filename, cluster_count):
        np.random.seed(9)
        self.cluster_count = int(cluster_count)
        self.dataset = []
        self._x_range, self._y_range = (1e9, -1e9), (1e9, -1e9)
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                [x, y] = line.split()
                x, y = float(x), float(y)
                self.dataset.append((x, y))

                self._x_range = (min(self._x_range[0], x), max(self._x_range[1], x))
                self._y_range = (min(self._y_range[0], y), max(self._y_range[1], y))

    def _get_starting_centroids(self):
        # uniform sampling
        xs = np.random.uniform(self._x_range[0], self._x_range[1], self.cluster_count)
        ys = np.random.uniform(self._y_range[0], self._y_range[1], self.cluster_count)
        return list(zip(xs, ys))

        # # kMeans++
        # centroids = random.sample(self.dataset, 1)
        # dists = [1e18] * len(self.dataset)
        # for i in range(self.cluster_count - 1):
        #     for j, _ in enumerate(self.dataset):
        #         x_dist = self.dataset[j][0] - centroids[-1][0]
        #         y_dist = self.dataset[j][1] - centroids[-1][1]
        #         dist = x_dist * x_dist + y_dist * y_dist
        #         if dist < dists[j]:
        #             dists[j] = dist
        #     probs = np.array(dists) / np.sum(dists)
        #     [next_cent] = np.random.choice(range(len(self.dataset)), p=probs, size=1)
        #     dists[next_cent] = 0
        #     centroids.append(self.dataset[next_cent])
        # return centroids

    def _iterate(self, centroids):
        new_centroids = [(0, 0)] * self.cluster_count
        counts = [0] * self.cluster_count
        errors = [0] * self.cluster_count

        for (x, y) in self.dataset:
            cluster, dist = 0, 1e9
            for idx, (x_cent, y_cent) in enumerate(centroids):
                dist_sqr = (x - x_cent) * (x - x_cent) + (y - y_cent) * (y - y_cent)
                new_dist = np.sqrt(dist_sqr)
                if new_dist < dist:
                    cluster = idx
                    dist = new_dist
            new_centroids[cluster] = (
                new_centroids[cluster][0] + x,
                new_centroids[cluster][1] + y,
            )
            errors[cluster] += dist * dist
            counts[cluster] += 1

        error = 0
        for idx, _ in enumerate(new_centroids):
            new_centroids[idx] = (
                new_centroids[idx][0] / counts[idx] if counts[idx] else 0,
                new_centroids[idx][1] / counts[idx] if counts[idx] else 0,
            )
            error += counts[idx] * errors[idx]

        new_centroids.sort()
        return new_centroids, error

    def plot(self, centroids):
        colors = []
        for (x, y) in self.dataset:
            cluster, dist = 0, 1e9
            for idx, (x_cent, y_cent) in enumerate(centroids):
                new_dist = np.sqrt(
                    (x - x_cent) * (x - x_cent) + (y - y_cent) * (y - y_cent)
                )
                if new_dist < dist:
                    cluster = idx
                    dist = new_dist
            colors.append(cluster)

        viz_data = np.array(self.dataset)
        x, y = viz_data.T
        print(centroids)
        plt.scatter(x, y, c=colors)

        x_cent, y_cent = np.array(centroids).T
        plt.scatter(x_cent, y_cent, c=range(self.cluster_count), marker="x")

        plt.show()

    def run(self):
        best_centroids, best_error = None, None
        for i in range(self.NUM_RESTARTS):
            centroids = self._get_starting_centroids()
            centroids.sort()
            while True:
                new_centroids, error = self._iterate(centroids)
                different = False
                for x1, x2 in zip(centroids, new_centroids):
                    if not math.isclose(x1[0], x2[0]) or not math.isclose(x1[1], x2[1]):
                        different = True
                        break

                if not different:
                    break

                centroids = new_centroids

            if best_error is None or error < best_error:
                best_centroids = centroids
                best_error = error

        self.plot(best_centroids)


if __name__ == "__main__":
    id3 = KMeans()
    id3.init(sys.argv[1], sys.argv[2])
    id3.run()
