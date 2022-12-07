from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from pymfe.mfe import MFE
from sklearn.metrics import confusion_matrix
from collections import OrderedDict, defaultdict, deque

from Helper import get_confusion_matrix_values

def get_number_features(X):
    return X.shape[1]


def get_number_instance(X):
    return X.shape[0]

def get_confusion_matrix_values(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print( tn, fp, fn, tp)
    print("shape y true", y_true.shape, y_pred. shape)
    nbr_instance= y_true.shape[0]
    print("alll", nbr_instance)
    return(tn/nbr_instance, fp/nbr_instance, fn/nbr_instance,tp/nbr_instance)




def f1_score_berechnen(X_rest, y_rest, clf):
    y_pred = clf.predict(X_rest)
    score = f1_score(y_rest, y_pred)
    tn, fp, fn, tp=get_confusion_matrix_values(y_rest, y_pred)

    return score, tn, fp, fn, tp



def get_balance(y):
    positive = np.where(y == 1)[0]
    negative = np.where(y == 0)[0]
    return min(len(positive), len(negative))/max(len(positive), len(negative))

def get_statistical(X, y):
    mfe = MFE(groups=["Statistical"])
    mfe.fit(X, y)
    ft = mfe.extract()
    return ft

def nfo_theoretic(X, y):
    mfe = MFE(groups=["Information-theoretic"])
    mfe.fit(X, y)
    ft = mfe.extract()

    return ft

def Model_based(X, y):
    mfe = MFE(groups=["Model-based"])
    mfe.fit(X, y)
    ft = mfe.extract()
    return ft

#xtract metafeatures from a pre-fitted machine learning model (from sklearn package),
# you can use the extract_from_model method without needing to use the training data

# Extract from model
def extract_from_model(clf):
    extractor = MFE()
    ft = extractor.extract_from_model(clf)
    return ft
def get_itemset(X,y):
    mfe = MFE(groups=["itemset"])
    mfe.fit(X, y)
    ft = mfe.extract()
    return ft

def get_meta_features(X,y):
    mfe = MFE(features=["attr_to_inst","freq_class","inst_to_attr", "nr_attr","nr_inst","nr_attr", "nr_class","median",  "nr_num", "cov", "eigenvalues","nr_cor_attr","nr_disc"  , "nr_outliers", "p_trace", "sd_ratio", "ns_ratio","mut_inf", "eq_num_attr", "class_conc","attr_ent","attr_conc"  ])
    mfe.fit(X,y)
    ft = mfe.extract()
    return ft

#######################new meta-feature#######################

def log_number_of_features( X):
    return np.log(get_number_features(X))

def log_number_of_instances( X):
        return np.log(get_number_instance(X))
def class_occurences( X, y):
    if len(y.shape) == 2:
        occurences = []
        for i in range(y.shape[1]):
            occurences.append(class_occurences(X, y[:, i]))
        return occurences
    else:
        occurence_dict = defaultdict(float)
        for value in y:
            occurence_dict[value] += 1
        return occurence_dict


def class_probability_std(X, y):
    occurence_dict = class_occurences(X,y) # kann class_occurences von hier extraiert werden
    if len(y.shape) == 2:
        stds = []
        for i in range(y.shape[1]):
            std = np.array(
                [occurrence for occurrence in occurence_dict[i].values()],
                dtype=np.float64
            )
            std = (std / y.shape[0]).std()
            stds.append(std)
        return np.mean(stds)
    else:
        occurences = np.array(
            [occurrence for occurrence in occurence_dict.values()], dtype=np.float64
        )
        return (occurences / y.shape[0]).std()





def class_probability_mean(X, y):
    occurence_dict = class_occurences(X,y)

    if len(y.shape) == 2:
        occurences = []
        for i in range(y.shape[1]):
            occurences.extend(
            [occurrence for occurrence in occurence_dict[i].values()]
            )
        occurences = np.array(occurences)
    else:
        occurences = np.array(
            [occurrence for occurrence in occurence_dict.values()], dtype=np.float64
        )
    return (occurences / y.shape[0]).mean()



def class_probability_max(X, y):
    occurences = class_occurences(X,y)
    max_value = -1

    if len(y.shape) == 2:
        for i in range(y.shape[1]):
            for num_occurences in occurences[i].values():
                if num_occurences > max_value:
                    max_value = num_occurences
    else:
        for num_occurences in occurences.values():
            if num_occurences > max_value:
                max_value = num_occurences
    return float(max_value) / float(y.shape[0])

def class_probability_min(X, y):
    occurences = class_occurences(X,y)

    min_value = np.iinfo(np.int64).max
    if len(y.shape) == 2:
        for i in range(y.shape[1]):
            for num_occurences in occurences[i].values():
                if num_occurences < min_value:
                    min_value = num_occurences
    else:
        for num_occurences in occurences.values():
            if num_occurences < min_value:
                min_value = num_occurences
    return float(min_value) / float(y.shape[0])






def dataset_ratio( X):
    return float(get_number_features(X)) / float(
        get_number_instance(X))


def log_dataset_ratio(X):
    return np.log(dataset_ratio( X))


def inverse_dataset_ratio( X):
    return float(get_number_instance(X)) / float(
        get_number_features(X)
    )


def log_inverse_dataset_ratio(X):
    return np.log(inverse_dataset_ratio( X))
def number_of_categorical_features( categorical_indicator):
    return np.sum([value == True for value in categorical_indicator])

def number_of_numeric_features( categorical_indicator):
    return np.sum([value == False for value in categorical_indicator])


def ratio_nominal_to_numerical(categorical_indicator):
    num_categorical = float(
        number_of_categorical_features( categorical_indicator)
    )
    num_numerical = float(
        number_of_numeric_features( categorical_indicator)
    )
    if num_numerical == 0.0:
        return 0.0
    else:
        return num_categorical / num_numerical


def ratio_numerical_to_nominal(categorical_indicator):
    num_categorical = float(
        number_of_categorical_features( categorical_indicator)
    )
    num_numerical = float(
        number_of_numeric_features( categorical_indicator)
    )
    if num_categorical == 0.0:
        return 0.0
    else:
        return num_numerical / num_categorical

####################################
def number_missing_values(X):
    missing = np.count_nonzero(np.isnan(X))
    return missing

def percentage_mv(X):
    return float(number_missing_values(X)) / float(
        X.shape[0] * X.shape[1]
    )

def missing_values(X):
    missing = pd.isna(X)
    return missing


def number_Features_mv(X):
    missing =missing_values(X)
    num_missing = missing.sum(axis=0)
    return float(np.sum([1 if num > 0 else 0 for num in num_missing]))


def number_instances_mv(X):
    missing = missing_values(X)
    num_missing = missing.sum(axis=1)
    return float(np.sum([1 if num > 0 else 0 for num in num_missing]))


def percentage_instances_mv(X):
    n_missing = number_instances_mv(X)
    n_total = get_number_instance(X)
    return float(n_missing / n_total)


def percentage_features_mv(X):
    n_missing = number_Features_mv(X)
    n_total = get_number_features(X)
    return float(n_missing / n_total)