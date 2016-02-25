import numpy as np
from classifier import LinearModel, KNN
from numpy.random import multivariate_normal as mvn
import matplotlib.pyplot as plt
from data_generator import DataGenerator


def plot_results(dataset, Y, titles):
    for i, y in enumerate(Y):
        class_0 = dataset[np.where(y == 0)]
        class_1 = dataset[np.where(y == 1)]
        plt.subplot(220 + i)
        plt.scatter(class_0[:, 0], class_0[:, 1], s=25.0, c='b', marker='o', alpha=0.5)
        plt.scatter(class_1[:, 0], class_1[:, 1], s=25.0, c='c', marker='o', alpha=0.5)
        plt.title(titles[i])
    plt.show()


def create_data(means, var=np.eye(2), data=100):
    data_list = [np.hstack((mvn(mean, var, data), np.zeros((data, 1)) + class_num))
               for class_num, mean in enumerate(means)]
    return data_list


def print_accuracy(results, labels):
    print "{0} / {1} number of samples correctly predicted".format(len(np.where(results == labels)[0]),
                                                                   len(labels))


def main():
    class_means = np.array([[0, 0], [2.5, 2.5]])
    class_variances = [np.eye(2), np.eye(2)]
    num_components = 10
    num_desired_points_per_class = 200
    class_0, class_1 = DataGenerator.generate_gaussian_mixture(class_means, class_variances, num_components,
                                                               num_desired_points_per_class)
    combined_dataset = np.vstack((class_0, class_1))
    combined_labels = np.hstack((np.zeros(num_desired_points_per_class, dtype=np.int32),
                                 np.ones(num_desired_points_per_class, dtype=np.int32)))
    ln = LinearModel()
    ln.fit(train_data=combined_dataset, labels=combined_labels)
    r_linear_model = ln.predict(test_data=combined_dataset, extra_params_dict={'threshold': 0.5})
    print_accuracy(r_linear_model, combined_labels)
    knn = KNN(num_neighbors=5)
    knn.fit(train_data=combined_dataset, labels=combined_labels)
    r_knn = knn.predict(test_data=combined_dataset)
    print_accuracy(r_knn, combined_labels)
    plot_results(combined_dataset, np.vstack((np.vstack((combined_labels[np.newaxis, :],
                                                       r_linear_model[np.newaxis, :])),
                                                       r_knn[np.newaxis, :])), titles=['gt', 'lin', 'knn'])
    return


if __name__ == '__main__':
    main()