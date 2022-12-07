import openml

import numpy as np
from sklearn.model_selection import train_test_split
import time
from pymfe.mfe import MFE

from Data_sampling import data_test_sampling, data_train_sampling
from Clf_list import random_molde
from Datasets_info import get_number_features, get_number_instance,f1_score_berechnen, get_balance, log_number_of_features,log_number_of_instances,class_occurences,class_probability_std, class_probability_mean, class_probability_max,class_probability_min, inverse_dataset_ratio,log_inverse_dataset_ratio, dataset_ratio, log_dataset_ratio,number_of_categorical_features,number_of_numeric_features,ratio_nominal_to_numerical,ratio_numerical_to_nominal
import time
from Helper import get_confusion_matrix_values
def unified_data_train (X_transformer, y, clf_list):
    """damit wir für jeder clf nich di gleiche sampling wählen +++nachteil es wird zeit consoming+++"""
    """X_sampling, y_sampling, X_rest, y_rest = data_train_sampling(X_transformer, y)

    y_big = y_sampling[100:]"""
    column_name= []
    X_new = np.zeros((100, len(clf_list)))
    for clf_i in range(len(clf_list)):

        X_sampling, y_sampling, X_rest, y_rest = data_train_sampling(X_transformer, y)
        """"fit_time=time.time()"""
        clf_list[clf_i].fit(X_sampling[:100], y_sampling[:100])
        """print(f"{time.time()-fit_time}fit time für classifier#######{clf_list[clf_i]}")"""
        X_new[:, clf_i] = clf_list[clf_i].predict_proba(X_sampling[100:])[:, 0]
        score, tn, fp, fn, tp = f1_score_berechnen(X_rest, y_rest, clf_list[clf_i])
        X_new = np.insert(X_new, X_new.shape[1], values=score, axis=1)
        column_name.append(f"f1_score__{clf_list[clf_i]}")
        X_new = np.insert(X_new, X_new.shape[1], values=tn, axis=1)
        column_name.append(f"TN __{clf_list[clf_i]}")
        X_new = np.insert(X_new, X_new.shape[1], values=fp, axis=1)
        column_name.append(f"FP__ {clf_list[clf_i]}")
        X_new = np.insert(X_new, X_new.shape[1], values=fn, axis=1)
        column_name.append(f"FN__ {clf_list[clf_i]}")
        X_new = np.insert(X_new, X_new.shape[1], values=tp, axis=1)
        column_name.append(f"TP __{clf_list[clf_i]}")
    features_names= clf_list+column_name
    y_big = y_sampling[100:]
    return X_new, y_big, features_names





def unified_data_test (X_transformer, y, clf_list):
    our_system_time=time.time()

    X_train, X_test, y_train, y_test = train_test_split(X_transformer, y, test_size=0.33,
                                                                                random_state=42, stratify=y)
    X_sampling, y_sampling, X_rest, y_rest = data_test_sampling(X_train, y_train)
    column_name = []
    print("all clf list", clf_list)
    X_new = np.zeros((X_test.shape[0], len(clf_list)))
    for clf_i in range(len(clf_list)):
        print("feature name", column_name)
        print ("mnin jet",clf_list[clf_i])

        clf_list[clf_i].fit(X_sampling, y_sampling)
        X_new[:, clf_i] = clf_list[clf_i].predict_proba(X_test)[:, 0]
        score, tn, fp, fn, tp = f1_score_berechnen(X_rest, y_rest, clf_list[clf_i])
        X_new = np.insert(X_new, X_new.shape[1], values=score,axis=1)
        column_name.append(f"f1_score__{clf_list[clf_i]}")
        X_new = np.insert(X_new, X_new.shape[1], values=tn, axis=1)
        column_name.append(f"TN__{clf_list[clf_i]}")
        X_new = np.insert(X_new, X_new.shape[1], values=fp, axis=1)
        column_name.append(f"FP__{clf_list[clf_i]}")
        X_new = np.insert(X_new, X_new.shape[1], values=fn, axis=1)
        column_name.append(f"FN__{clf_list[clf_i]}")
        X_new = np.insert(X_new, X_new.shape[1], values=tp, axis=1)
        column_name.append(f"TP__{clf_list[clf_i]}")



    features_names= clf_list+ column_name
    """ft = get_meta_features(X_transformer, y)
    for x, y in zip(ft[0], ft[1]):
        X_new = np.insert(X_new, X_new.shape[1], values=y, axis=1)"""
    #X_new=np.insert(X_new, X_new.shape[1], values=get_number_features(X_transformer), axis=1)

    #X_new=np.insert(X_new, X_new.shape[1], values=get_number_instance(X_transformer), axis=1)

    print("time consoming our system==================",time.time()-our_system_time)
    return X_new, y_test, features_names


