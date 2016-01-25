import numpy as np
from networkx.algorithms.bipartite import projection
import random
from numpy.random import multivariate_normal as mvn
import matplotlib.pyplot as plt


def plot_data(dataset, colors):
    for data, color in zip(dataset, colors):
        plt.scatter(data[:, 0], data[:, 1], len(data), c=color)
    plt.show()


def create_data(means, var=np.eye(2), data=100):
    dataset = [np.hstack((mvn(mean, var, data), np.zeros((data, 1)) + class_num))
               for class_num, mean in enumerate(means)]
    return dataset

def fit_linear_model(dataset):
    return

def main():
    colors = ['b', 'r']
    means = np.array([[0, 0], [2.5, 2.5]])
    dataset = create_data(means)
    plot_data(dataset, colors)
    combined_dataset = np.vstack((dataset[0], dataset[1]))
    np.random.shuffle(combined_dataset)
    return

if __name__ == '__main__':
    main()