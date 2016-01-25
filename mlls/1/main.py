import numpy as np
from sklearn import linear_model
from numpy.random import multivariate_normal as mvn
import matplotlib.pyplot as plt


def plot_results(dataset, Y, titles):
    for i, y in enumerate(Y):
        class_0 = dataset[np.where(y == 0)]
        class_1 = dataset[np.where(y == 1)]
        plt.subplot(220 + i)
        plt.scatter(class_0[:, 0], class_0[:, 1], len(class_0), c='b')
        plt.scatter(class_1[:, 0], class_1[:, 1], len(dataset), c='r')
        plt.title(titles[i])
    plt.show()


def create_data(means, var=np.eye(2), data=100):
    data_list = [np.hstack((mvn(mean, var, data), np.zeros((data, 1)) + class_num))
               for class_num, mean in enumerate(means)]
    return data_list


def print_accuracy(results, dataset):
    print "{0} / {1} number of samples correctly predicted".format(len(np.where(results == dataset[:, -1])[0]),
                                                                   len(dataset))


def get_nearest_neighbors(k, index, dataset):
    euclidean_dist = [np.linalg.norm(dataset[curr_index, :2] - dataset[index, :2])
                      for curr_index, curr_element in enumerate(dataset)
                      if curr_index != index]
    sorted_args = np.argsort(euclidean_dist)[:k]
    return np.array([dataset[arg] for arg in sorted_args], np.int32)


def get_class_pred(k, index, dataset):
    nbrs = get_nearest_neighbors(k, index, dataset)
    return np.bincount(nbrs[:, -1]).argmax()


def knn(k, dataset):
    return np.array([get_class_pred(k, index, dataset) for index, _ in enumerate(dataset)], np.int32)


def predict_linear_model(W, dataset, threshold=0.5):
    return np.array([0 if val < threshold else 1 for val in dataset.dot(W)], np.int32)


def fit_linear_model(dataset):
    X, Y = np.hstack((np.ones((len(dataset), 1)), dataset[:, :-1])), dataset[:, -1]
    W = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
    return W


def main():
    means = np.array([[0, 0], [2.5, 2.5]])
    data_list = create_data(means)
    combined_dataset = np.vstack((data_list[0], data_list[1]))
    r_linear_model = predict_linear_model(fit_linear_model(combined_dataset), combined_dataset)
    print_accuracy(r_linear_model, combined_dataset)
    r_knn = knn(5, combined_dataset)
    print_accuracy(r_knn, combined_dataset)
    plot_results(combined_dataset[:, :-1], np.vstack((np.vstack((combined_dataset[:, -1][np.newaxis, :],
                                                        r_linear_model[np.newaxis, :])),
                                                        r_knn[np.newaxis, :])), titles=['gt', 'lin', 'knn'])
    return


if __name__ == '__main__':
    main()