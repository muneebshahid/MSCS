from data_generator import DataGenerator as dg
import numpy as np
from rank_features import CorrelationCoefficient, Relief
from sklearn.neighbors import KNeighborsClassifier


def remove_features(data, features_indices, drop_indices=2):
    rows = len(data)
    cols = len(data[0])
    removed_features = np.empty((rows, cols-drop_indices))
    for i in range(len(features_indices[:-2])):
        removed_features[:, i] = data[:, features_indices[i]]
    return removed_features

def split_train_test(data, split_num=None, split_ratio=0.7):
    train_size = int(np.ceil(len(data) * split_ratio)) if split_num is None else split_num
    return data[:train_size], data[train_size:]

def main():
    init_means = [-1, 1]
    mean_dimensions = 10
    points_per_class = 250
    knc = KNeighborsClassifier(n_neighbors=5)
    means = [[init_mean for mean_dim in range(mean_dimensions)]for init_mean in init_means]
    variances = [np.eye(len(means[0])), np.eye(len(means[0]))]
    data = dg.generate_prob_mixture(class_means=means, class_variances=variances, num_components=5,
                                    num_desired_points=points_per_class, dim_uniform=2)
    class_0 = np.hstack((data[0], np.zeros((len(data[0]), 1))))
    class_1 = np.hstack((data[1], np.ones((len(data[0]), 1))))
    train_data_class_0, test_data_class_0 = split_train_test(class_0)
    train_data_class_1, test_data_class_1 = split_train_test(class_1)
    train_data = np.vstack((train_data_class_0, train_data_class_1))
    test_data = np.vstack((test_data_class_0, test_data_class_1))
    corr_ranked_features, _ = CorrelationCoefficient.rank_features(train_data[:, :-1], train_data[:, -1])
    relief_ranked_features, _ = Relief.rank_features(train_data[:, :-1], train_data[:, -1])

    knc.fit(train_data[:, :-1], train_data[:, -1])
    pred = pred_test_default = knc.predict(test_data[:, :-1])
    print len(np.where(pred != test_data[:, -1])[0])

    corr_train_removed_features = remove_features(train_data[:, :-1], corr_ranked_features)
    corr_test_removed_features = remove_features(test_data[:, :-1], corr_ranked_features)
    knc.fit(corr_train_removed_features, train_data[:, -1])
    pred = knc.predict(corr_test_removed_features)
    print len(np.where(pred != test_data[:, -1])[0])

    corr_train_removed_features = remove_features(train_data[:, :-1], relief_ranked_features)
    corr_test_removed_features = remove_features(test_data[:, :-1], relief_ranked_features)
    knc.fit(corr_train_removed_features, train_data[:, -1])
    pred = knc.predict(corr_test_removed_features)
    print len(np.where(pred != test_data[:, -1])[0])
    return

if __name__ == '__main__':
    main()