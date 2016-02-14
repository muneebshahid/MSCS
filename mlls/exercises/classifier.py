import abc


class Classifier:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """
        :return:
        """

    @abc.abstractmethod
    def fit(self, train_data):
        """
        :param train_data: Trains the classifier by fitting the train_data
        :return: None
        """

    @abc.abstractmethod
    def predict(self, test_data):
        """
        :param test_data: returns predictions on test_data
        :return:
        """


class KNN(Classifier):

    def __init__(self):
        super(KNN, self).__init__()
        return
    
    def fit(self, train_data):
        return

    def predict(self, test_data):
        return


class LinearModel(Classifier):

    def __init__(self):
        super(LinearModel, self).__init__()
        return

    def fit(self, train_data):
        return

    def predict(self, test_data):
        return