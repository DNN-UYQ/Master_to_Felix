import numpy as np
# It is a function, that must extract a balanced sampling from ech dataset
def data_sampling(X_list, y_list):
    positive = np.where(y_list == 1)[0]
    negative = np.where(y_list == 0)[0]
    random_id_positive = np.random.permutation(positive)
    random_id_negative = np.random.permutation(negative)
    X_sampling = []
    y_sampling= []
    for i in range(100):
        X_sampling.append(X_list[random_id_positive[i]])
        y_sampling.append(y_list[random_id_positive[i]])
        X_sampling.append(X_list[random_id_negative[i]])
        y_sampling.append(y_list[random_id_negative[i]])

    return np.array(X_sampling), np.array(y_sampling)