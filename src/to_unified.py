import openml
import numpy as np
from Clf_list import random_molde
from Datasets_info import get_number_features, get_number_instance, f1_score_berechnen
def unified_data(X_sampling, y_sampling, clf_list, number_features, number_instances):
    y_big = []


    X_new = np.zeros((100, len(clf_list)))
    y_big.append(y_sampling[100:])
    for clf_i in range(len(clf_list)):
        f1_score_features=[]
        clf_list[clf_i].fit(X_sampling[:100], y_sampling[:100])
        X_new[:, clf_i] = clf_list[clf_i].predict_proba(X_sampling[100:])[:, 0]
        score = f1_score_berechnen(X_sampling, y_sampling, clf_list[clf_i])
        f1_score_features.append(f"F1_score:{clf_list[clf_i]}")
        X_new = np.insert(X_new, X_new.shape[1], values=score, axis=1)

    X_new=np.insert(X_new, X_new.shape[1], values=number_features, axis=1)

    X_new=np.insert(X_new, X_new.shape[1], values=number_instances, axis=1)


    y_big = np.array(y_big).reshape((len(y_big)*100, ))


    return X_new, y_big
