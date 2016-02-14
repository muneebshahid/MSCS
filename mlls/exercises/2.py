import numpy as np
from numpy.random import multivariate_normal as mvn
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


def point_in_rectangle(rectangle, point):
    return rectangle[0][0] < point[0] < rectangle[2][0] and rectangle[0][1] < point[1] < rectangle[2][1]


def get_points_in_rectangle(rectangle, data):
    pts_in_rect = [[], []]
    for instance in data:
        if point_in_rectangle(rectangle, instance[:-1]):
            instance_class = int(instance[-1])
            pts_in_rect[instance_class].append(instance)
    return pts_in_rect


def calc_prior(pts_in_rect):
    len_class_0, len_class_1 = len(pts_in_rect[0]), len(pts_in_rect[1])
    assert len_class_0 != 0
    assert len_class_1 != 1
    len_total = float(len_class_0 + len_class_1)
    return len_class_0 / len_total, len_class_1 / len_total


def calc_posterior(means, data):
    prior = calc_prior(data)
    posterior = [[], []]
    mvn_pdf = [multivariate_normal(mean=means[0], cov=np.eye(2)), multivariate_normal(mean=means[1], cov=np.eye(2))]
    for i, mean in enumerate(means):
        posterior[i] = 0
        for point in data[i]:
            #for mean
            point_coordinates = point[:-1]
            likelihood = mvn_pdf[i].pdf(point_coordinates)
            numerator = likelihood * prior[i]
            normalizer = mvn_pdf[0].pdf(point_coordinates) * prior[0] + mvn_pdf[1].pdf(point_coordinates) * prior[1]
            posterior[i] += (numerator / normalizer)
    return posterior


def main():
    # create centers
    means = [[1.0, 1.0], [2.5, 2.5]]
    data = []
    rect_start_x = 2.0
    rect_start_y = 2.0
    rect_width = 1.0
    rec_height = 1.0
    for i, mean in enumerate(means):
        class_centers = mvn(mean, np.eye(2), 10)
        class_data = [np.hstack((mvn(class_center, np.eye(2), 10), np.zeros((10, 1)) + i))
                               for class_center in class_centers]
        data.append(class_data)
    data = np.array(data).reshape(200, 3)
    rectangle = np.array([[rect_start_x, rect_start_y],
                          [rect_start_x + rect_width, rect_start_y],
                          [rect_start_x + rect_width, rect_start_y + rec_height],
                          [rect_start_x, rect_start_y + rec_height],
                          [rect_start_x, rect_start_y]]).reshape(5, 2)

    pts = get_points_in_rectangle(rectangle, data)
    posterior = calc_posterior(means, pts)
    print posterior
    for class_data in data:
        plt.plot(class_data[:, 0], class_data[:, 1], 'o', markersize=8)
    plt.plot(rectangle[:, 0], rectangle[:, 1])
    plt.show()

if __name__ == '__main__':
    main()