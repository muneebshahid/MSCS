from data_generator import DataGenerator as dg
import numpy as np
from classifier import KNN
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def split_train_test(data, split_num=None, split_ratio=0.7):
    train_size = int(np.ceil(len(data) * split_ratio)) if split_num is None else split_num
    return data[:train_size], data[train_size:]


def main():
    means = [[-1, -1], [1.0, 1.0]]
    variances = [np.random.rand]
    knn_models = [3, 5, 10]
    data_sizes = [10, 25, 50, 75, 100, 125, 150, 175, 200]
    points_per_class = 500
    data = dg.generate_gaussian_mixture(class_means=means, class_variances=np.eye(2),
                                        num_components=5, num_desired_points_per_class=points_per_class)
    class_0 = np.hstack((data[0], np.zeros((len(data[0]), 1))))
    class_1 = np.hstack((data[1], np.ones((len(data[0]), 1))))
    results_train = np.empty((len(knn_models), len(data_sizes)))
    results_test = np.empty((len(knn_models), len(data_sizes)))
    train_data_class_0, test_data_class_0 = split_train_test(class_0)
    train_data_class_1, test_data_class_1 = split_train_test(class_1)
    print 'train size, test size', len(train_data_class_1), len(test_data_class_1)
    train_data = np.vstack((train_data_class_0, train_data_class_1))
    test_data = np.vstack((test_data_class_0, test_data_class_1))
    for i, knn_model in enumerate(knn_models):
        kncs = KNeighborsClassifier(n_neighbors=knn_model)
        for j, data_size in enumerate(data_sizes):
            curr_train_class_0, curr_train_class_1 = train_data_class_0[:data_size], train_data_class_1[:data_size]
            curr_train_data = np.vstack((curr_train_class_0, curr_train_class_1))
            kncs.fit(curr_train_data[:, :2], curr_train_data[:, -1])
            predictions_train = kncs.predict(train_data[:, :2])
            predictions_test = kncs.predict(test_data[:, :2])
            results_train[i][j] = len(np.where(predictions_train != train_data[:, -1])[0]) / float(len(train_data))
            results_test[i][j] = len(np.where(predictions_test != test_data[:, -1])[0]) / float(len(test_data))

    plt.plot(data_sizes, results_test[0, :], 'r')
    plt.plot(data_sizes, results_test[1, :], 'b')
    plt.plot(data_sizes, results_test[2, :], 'g')
    plt.plot(data_sizes, results_train[0, :], 'r--')
    plt.plot(data_sizes, results_train[1, :], 'b--')
    plt.plot(data_sizes, results_train[2, :], 'g--')
    plt.show()
if __name__ == '__main__':
    main()