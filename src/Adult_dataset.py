from sklearn.model_selection import train_test_split
from Preprocessing import data_download, data_sampling
from Clf_list import classfier
import numpy as np
def adult_dataset():
    X_adult, y_adult = data_download(data_id=[179])

    X_train_adult, X_test_adult, y_train_adult, y_test_adult = train_test_split(X_adult[0], y_adult[0], test_size=0.33,
                                                        random_state=42, stratify=y_adult[0])
    return X_train_adult, X_test_adult, y_train_adult, y_test_adult





def featurize(X_train_adult, y_train_adult, X_test_adult, clf_list):
    X_train_adult, y_train_adult = data_sampling(X_train_adult, y_train_adult)
    X_big_adult = []
    num_rows = X_test_adult.shape[0]
    X_test_new = np.zeros((num_rows, len(clf_list)))
    for clf_i in range(len(clf_list)):
        clf_list[clf_i].fit(X_train_adult, y_train_adult)
        X_test_new[:, clf_i] = clf_list[clf_i].predict_proba(X_test_adult)[:, 0]
    X_big_adult.append(X_test_new)
    X_big_adult= np.squeeze(X_big_adult, axis=0)
    return X_big_adult


