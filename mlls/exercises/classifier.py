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
    def fit(self, train_data, labels, extra_params_dict=None):
        """
        :param train_data: Trains the classifier by fitting the train_data
        :param labels
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
    __labels = None

    def __init__(self, num_neighbors):
        super(KNN, self).__init__()
        self.__num_neighbors = num_neighbors
        return
    
    def fit(self, train_data, labels, extra_params_dict=None):
        self.__train_data = train_data
        self.__labels = labels
        return

    def predict(self, test_data, extra_params_dict=None):
        return np.array([self.__get_class_prediction(test_instance) for test_instance in test_data], np.int32)

    def __get_class_prediction(self, instance):
        nbrs = self.__get_nearest_neighbors(instance)
        return np.bincount(nbrs).argmax()

    def __get_nearest_neighbors(self, instance):
        euclidean_dist = [np.linalg.norm(curr_element - instance) for curr_element in self.__train_data]
        sorted_args = np.argsort(euclidean_dist)[:self.__num_neighbors]
        return np.array([self.__labels[arg] for arg in sorted_args], np.int32)


class LinearModel(Classifier):

    __weights = None
    __train_data = None
    __labels = None

    def __init__(self):
        super(LinearModel, self).__init__()
        return

    def fit(self, train_data, labels, extra_params_dict=None):
        ridge_param = 0 if extra_params_dict is None else extra_params_dict['ridge_param']
        X, Y = train_data, labels
        ridge_matrix = np.eye(np.shape(X.T)[0]).dot(ridge_param)
        self.__weights = np.linalg.inv(np.add(X.T.dot(X), ridge_matrix)).dot(X.T.dot(Y))
        return self.__weights

    def predict(self, test_data, extra_params_dict=None):
        return np.array([0 if val < extra_params_dict['threshold'] else 1
                         for val in test_data.dot(self.__weights)], np.int32)
