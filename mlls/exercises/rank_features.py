import abc
import numpy as np

class RankFeatures:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """
        :return:
        """
    @staticmethod
    @abc.abstractmethod
    def rank_features(data, labels):
        """
        :param data:
        :param labels:
        :return:
        """


class CorrelationCoefficient(RankFeatures):

    def __init__(self):
        super(CorrelationCoefficient, self).__init__()

    @staticmethod
    def rank_features(data, labels):
        corr_coef_ranking = []
        for feature in range(len(data[0])):
            curr_feature = data[:, feature]
            corr_coef_ranking.append(np.corrcoef(curr_feature, labels)[0][1])
        return np.argsort(corr_coef_ranking)[::-1], corr_coef_ranking


class Relief(RankFeatures):

    def __init__(self):
        super(Relief, self).__init__()

    @staticmethod
    def rank_features(data, labels):
        sep_classes = data[np.where(labels == 0)], data[np.where(labels == 1)]
        T = 20
        t = 0
        w = np.zeros(len(data[0]))
        while t < T:
            class_index, instance_index = np.random.random_integers(0, 1), np.random.randint(0, len(sep_classes[0]))
            x = sep_classes[class_index][instance_index]
            mod_array = np.delete(sep_classes[class_index].copy(), instance_index, 0)
            nh = Relief.__get_closest_feature(x, mod_array)
            nm = Relief.__get_closest_feature(x, sep_classes[abs(class_index - 1)])
            for i in range(len(w)):
                w[i] = w[i] + np.abs(x[i] - nm[i]) - np.abs(x[i] - nh[i])
            t += 1
        normalize_w = w / np.linalg.norm(w)
        return np.argsort(normalize_w)[::-1], w

    @staticmethod
    def __get_closest_feature(x, data):
        feature_distances = [np.linalg.norm(x - curr_feature) for curr_feature in data]
        closest_feature = data[np.argsort(feature_distances)[0]]
        return closest_feature
