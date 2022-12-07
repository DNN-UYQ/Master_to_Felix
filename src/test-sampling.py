import numpy as np
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from sklearn.model_selection import train_test_split

import autosklearn.classification
import pandas as pd
from autosklearn.metrics import f1
from Load_datasets import load_unified_data
import time
from Permutation_importance import per_import
import sklearn
from Helper import load_list
import numpy as np
from Baseline import baseline
from sklearn.inspection import plot_partial_dependence, permutation_importance
import matplotlib.pyplot as plt
import os
import openml
from collections import OrderedDict, defaultdict, deque
import scipy.sparse
import scipy.stats
from scipy.linalg import LinAlgError
from scipy.sparse import csr_matrix
from autosklearn.util import logging_
from Datasets_info import get_number_features,get_number_instance

import pandas as pd



from Numeric_transformer import transformer
from Load_datasets import load_unified_data



"""X_train, y_train = load_unified_data("ilyes_new.xlsx")
print(X_train.shape)
print(y_train.shape)
print("##############################")"""

X_test= np.load("all_x_test.npy", allow_pickle=True).item()
y_test= np.load("all_y_test.npy", allow_pickle=True).item()
for key in X_test:
    print(key)
    print(y_test[key].shape)
    print(X_test[key].shape)

"""automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=600,metric=f1)
time_our_methode=time.time()                                                                       
automl.fit(X_train, y_train)
#### our methode fitting #######
print("our methode fitting",time.time_ns() - time_our_methode)
model_score = automl.score(X_train, y_train)
print(f"automl model score {model_score}")

#### our methode test phase #######
for key in X_test:
    predictions = automl.predict(X_test[key])
    print(f"f1 score für test dataset##{key}##", sklearn.metrics.f1_score(y_test[key], predictions))"""

#### Besline auto-sklear #########
for dataset in [31, 41159, 1461, 40981,1590]:
    print(f"data set id key {dataset}")
    test_dataset = openml.datasets.get_dataset(dataset_id=dataset)
    X, y, categorical_indicator, attribute_names = test_dataset.get_data(dataset_format="array",
                                                                         target=test_dataset.default_target_attribute)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=42, stratify=y)
    for suchzeit in [30, 60, 120]:

        auto_model = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=suchzeit)
        auto_model.fit(X_train,y_train)
        pred = auto_model.predict(X_test)
        print("f11 auto sklearn model testing", sklearn.metrics.f1_score(pred, y_test))
        print("zeit", suchzeit)