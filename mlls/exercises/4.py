from data_generator import DataGenerator as dg
import numpy as np
from classifier import KNN
from sklearn.neighbors import KNeighborsClassifier as knn_classifier_scipy

def split_train_test(data, split_ratio=0.7):
    np.random.shuffle(data)
    train_size = int(np.ceil(len(data) * split_ratio))
    return data[:train_size], data[train_size:]


def main():
    means = [[0, 0], [2.5, 2.5]]
    data = dg.generate_gaussian_mixture(class_means=means, class_variance=np.eye(2),
                                        num_components=5, num_desired_points_per_class=10)
    train_data, test_data = split_train_test(data)
    knn_classifier_scipy.fit(train_data, )

if __name__ == '__main__':
    main()