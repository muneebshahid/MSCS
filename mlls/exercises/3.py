import numpy as np
from numpy.random import multivariate_normal as mvn
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knc


def split(data):
    np.random.shuffle(data)
    return data[:150], data[150:]


def bias_variance(data, nbrs, bootstrap=10):
    result_dict = {}
    for bootsrap_num in bootstrap:
        train, test = split(data)
        classifier = knc(n_neighbors=nbrs)
        coordinates_train, labels_train = train[:, :2], train[:, -2]
        classifier.fit(coordinates_train, labels_train)
        for instance in test:
            coordinates_test = instance[:2]
            pred = classifier.predict(coordinates_test)
            result_dict[instance[:-1]].append(pred)



def main():
    # create centers
    means = [[0, 0], [2.5, 2.5]]
    data = []
    for class_num, mean in enumerate(means):
        class_centers = mvn(mean, np.eye(2), 10)
        class_data = [np.hstack((mvn(class_center, np.eye(2), 10), np.zeros((10, 1)) + class_num))
                               for class_center in class_centers]
        data.extend(class_data)
    data = np.array(data).reshape(200, 3)
    data = np.hstack((data, np.zeros((200, 1))))
    bias_variance(data, 2)


if __name__ == '__main__':
    main()