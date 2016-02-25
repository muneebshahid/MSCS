import numpy as np
from classifier import LinearModel, KNN
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

def main():
    means = np.array([[0, 0], [2.5, 2.5]])
    data_list = create_data(means)
    combined_dataset = np.vstack((data_list[0], data_list[1]))
    ln = LinearModel()
    ln.fit(train_data=combined_dataset)
    r_linear_model = ln.predict(test_data=combined_dataset, extra_params_dict={'threshold': 0.5})
    print_accuracy(r_linear_model, combined_dataset)
    knn = KNN(num_neighbors=5)
    knn.fit(train_data=combined_dataset)
    r_knn = knn.predict(test_data=combined_dataset[:, :-1])
    print_accuracy(r_knn, combined_dataset)
    #plot_results(combined_dataset[:, :-1], np.vstack((np.vstack((combined_dataset[:, -1][np.newaxis, :],
    #                                                    r_linear_model[np.newaxis, :])),
    #                                                    r_knn[np.newaxis, :])), titles=['gt', 'lin', 'knn'])
    return


if __name__ == '__main__':
    main()