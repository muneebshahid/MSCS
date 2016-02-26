import numpy as np
from classifier import LinearModel

def load_data(file_path):
    data = None
    with open(file_path) as file_handle:
        data = [line.split() for line in file_handle.readlines() if line.split()]
    return data


def main():
    limit=20
    folder_path = '../data/cancer micro array data/'
    train_data_file = folder_path + '14cancer.xtrain'
    train_labels_file = folder_path + '14cancer.ytrain'
    test_data_file = folder_path + '14cancer.xtest'
    test_labels_file = folder_path + '14cancer.ytest'
    train_data = np.array(load_data(train_data_file), np.float64).T[:limit]
    train_labels = np.array(load_data(train_labels_file), np.float64).T[:limit]
    test_data = np.array(load_data(test_data_file), np.float64).T
    test_labels = np.array(load_data(test_labels_file), np.float64).T
    print train_data.shape, train_labels.shape, test_data.shape
    ln = LinearModel()
    ln.fit(train_data=train_data, labels=train_labels, extra_params_dict={'ridge_param':0.001})
    ln.predict(test_data=test_data)
    return

if __name__ == '__main__':
    main()