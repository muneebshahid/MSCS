import abc
import numpy as np


class Classifier:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """
        :return:
        """

    @abc.abstractmethod
    def fit(self, train_data, extra_params_dict=None):
        """
        :param train_data: Trains the classifier by fitting the train_data
        :param extra_params_dict: optional extra params that the function might require
        :return: None
        """

    @abc.abstractmethod
    def predict(self, test_data, extra_params_dict=None):
        """
        :param test_data: returns predictions on test_data
        :param extra_params_dict: optional extra params that the function might require
        :return:
        """


class KNN(Classifier):

    __num_neighbors = None
    __train_data = None

    def __init__(self, num_neighbors):
        super(KNN, self).__init__()
        self.__num_neighbors = num_neighbors
        return
    
    def fit(self, train_data, extra_params_dict=None):
        self.__train_data = train_data
        return

    def predict(self, test_data, extra_params_dict=None):
        return np.array([self.__get_class_prediction(test_instance) for test_instance in test_data], np.int32)

    def __get_class_prediction(self, instance):
        nbrs = self.__get_nearest_neighbors(instance)
        return np.bincount(nbrs[:, -1]).argmax()

    def __get_nearest_neighbors(self, instance):
        euclidean_dist = [np.linalg.norm(curr_element[:2] - instance) for curr_element in self.__train_data]
        sorted_args = np.argsort(euclidean_dist)[:self.__num_neighbors]
        return np.array([self.__train_data[arg] for arg in sorted_args], np.int32)


class LinearModel(Classifier):

    __weights = None

    def __init__(self):
        super(LinearModel, self).__init__()
        return

    def fit(self, train_data, extra_params_dict=None):
        X, Y = np.hstack((np.ones((len(train_data), 1)), train_data[:, :-1])), train_data[:, -1]
        self.__weights = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
        return self.__weights

    def predict(self, test_data, extra_params_dict=None):
        """
        :param test_data:
        :param extra_params_dict: expects threshold as an extra param
        :return:
        """
        return np.array([0 if val < extra_params_dict['threshold'] else 1
                         for val in test_data.dot(self.__weights)], np.int32)