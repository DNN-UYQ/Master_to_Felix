import numpy as np
# It is a function, that must extract a balanced sampling from ech dataset
def data_train_sampling(X, y):
    positive = np.where(y == 1)[0]
    negative = np.where(y == 0)[0]
    random_id_positive = np.random.permutation(positive)
    random_id_negative = np.random.permutation(negative)
    print(random_id_positive.shape)
    print(random_id_negative.shape)


    X_sampling = []
    y_sampling= []
    X_rest=[]
    y_rest=[]
    remove_idx=[]
    for i in range(100):
        X_sampling.append(X[random_id_positive[i]])
        y_sampling.append(y[random_id_positive[i]])
        remove_idx.append(random_id_positive[i])
        X_sampling.append(X[random_id_negative[i]])
        y_sampling.append(y[random_id_negative[i]])
        remove_idx.append(random_id_negative[i])

    for idx, element in enumerate(X):

        # checking if element not present in index list
        if idx not in remove_idx:
            X_rest.append(element)
            y_rest.append(y[idx])
    return np.array(X_sampling), np.array(y_sampling), np.array(X_rest), np.array(y_rest)

def data_test_sampling(X, y):

    positive = np.where(y == 1)[0]
    negative = np.where(y == 0)[0]
    random_id_positive = np.random.permutation(positive)
    random_id_negative = np.random.permutation(negative)
    X_sampling = []
    y_sampling= []
    X_rest=[]
    y_rest=[]
    remove_idx=[]
    for i in range(50):
        X_sampling.append(X[random_id_positive[i]])
        y_sampling.append(y[random_id_positive[i]])
        remove_idx.append(random_id_positive[i])
        X_sampling.append(X[random_id_negative[i]])
        y_sampling.append(y[random_id_negative[i]])
        remove_idx.append(random_id_negative[i])

    for idx, element in enumerate(X):

        # checking if element not present in index list
        if idx not in remove_idx:
            X_rest.append(element)
            y_rest.append(y[idx])
    return np.array(X_sampling), np.array(y_sampling), np.array(X_rest), np.array(y_rest)
