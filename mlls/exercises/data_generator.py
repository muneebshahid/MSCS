import numpy as np
from numpy.random import multivariate_normal as mvn_np
from numpy.random import uniform


class DataGenerator:

    def __init__(self):

        return

    @staticmethod
    def generate_prob_mixture(class_means, class_variances, num_components,
                              num_desired_points, dim_uniform):
        data = []
        uniform_data = None
        normal_data = DataGenerator.generate_gaussian_mixture(class_means, class_variances, num_components,
                                                              num_desired_points)
        if dim_uniform > 0:
            uniform_data = DataGenerator.generate_uniform(-50, 50, range(len(class_means)), num_desired_points, dim_uniform)
        for i, class_mean in enumerate(class_means):
            if uniform_data is not None:
                normal_data[i][:, -dim_uniform:] = uniform_data[i]
            data.append(normal_data[i])
        return data

    @staticmethod
    def generate_gaussian_mixture(class_means, class_variances, num_components, num_desired_points_per_class, seed=42):
        data = []
        #np.random.seed(seed)
        num_points_per_component = num_desired_points_per_class / num_components
        for class_num, (mean, variance) in enumerate(zip(class_means, class_variances)):
            class_centers = mvn_np(mean, variance, num_components)
            class_data = [mvn_np(class_center, variance, num_points_per_component)
                          for class_center in class_centers]
            data.append(np.array(class_data).reshape(num_desired_points_per_class, len(class_means[0])))
        return data

    @staticmethod
    def generate_uniform(min, max, num_classes, num_points, dim):
        data = []
        for data_class in num_classes:
            data.append(uniform(min, max, (num_points, dim)))
        return data
