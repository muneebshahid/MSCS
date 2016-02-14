import numpy as np
from numpy.random import multivariate_normal as mvn_np


class DataGenerator:

    def __init__(self):
        return

    @staticmethod
    def generate_gaussian_mixture(class_means, class_variance, num_components, num_desired_points_per_class):
        data = []
        num_points_per_component = num_desired_points_per_class / num_components
        for class_num, mean in enumerate(class_means):
            class_centers = mvn_np(mean, class_variance, num_components)
            class_data = [np.hstack((mvn_np(class_center, class_variance, num_points_per_component),
                          np.zeros((num_points_per_component, 1)) + class_num))
                          for class_center in class_centers]
            data.extend(class_data)
        data = np.array(data).reshape(len(class_means) * num_desired_points_per_class, 3)
        return data
