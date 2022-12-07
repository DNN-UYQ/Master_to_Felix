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


import warnings
warnings.simplefilter(action='ignore')

from sklearn.metrics import accuracy_score



if __name__ == '__main__':
    start_time=time.time()
    #unified data to train our automl model
    X_train, y_train= load_unified_data("evaluation_200_clf.xlsx")             ######anderung
    #unified data to test our automl model
    #X_test, y_test = load_unified_data("data_test1.xlsx")
    X_test= np.load("evalX_200_clf.npy", allow_pickle=True).item()             ######anderung
    y_test= np.load("evaly_200_clf.npy", allow_pickle=True).item()             ######anderung


    automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=600,metric=f1)
    time_our_methode=time.time()
    automl.fit(X_train, y_train)
    print("our methode tiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiim",time.time() - time_our_methode)
    model_score = automl.score(X_train, y_train)
    print(f"automl model score {model_score}")
    Data_id = []
    F1_sc = []
    avr_result = 0
    for key in X_test:
        Data_id.append(key)
        test_time = time.time()
        predictions = automl.predict(X_test[key])
        f1_rechner = sklearn.metrics.f1_score(y_test[key], predictions)
        avr_result = avr_result + f1_rechner
        print("tessssssssssssssssssssssssssssssssst", time.time() - test_time)
        F1_sc.append(f1_rechner)
        print(f"f1 score für test dataset##{key}##", f1_rechner)
    Data_id.append("result")
    F1_sc.append(avr_result / len(X_test))
    result = {"data_id": Data_id, "f1_score": F1_sc}
    df = pd.DataFrame(result)
    df.to_excel("result_test_200_clf.xlsx")








    """r = permutation_importance(automl, X_train, y_train, n_repeats=10, random_state=42)
    sort_idx = r.importances_mean.argsort()[::-1]
    print(r)
    print(sort_idx)"""




    """new_automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=600, metric=f1)

    new_dataset = np.zeros((X_train.shape[0], 30))
    for index_feature in range(30):
        new_dataset[:, index_feature] = X_train[:, sort_idx[index_feature]].T

    new_automl.fit(new_dataset, y_train)



    Data_id = []
    F1_sc = []
    avr_result = 0
    for key in X_test:
        dataset=X_test[key]
        test_dataset = np.zeros((dataset.shape[0], 30))
        for index_feature in range(30):
            test_dataset[:, index_feature] = dataset[:, sort_idx[index_feature]].T
        Data_id.append(key)
        test_time = time.time()
        print("shapeeee",test_dataset.shape)


        predictions = new_automl.predict(test_dataset)
        f1_rechner = sklearn.metrics.f1_score(y_test[key], predictions)
        avr_result = avr_result + f1_rechner
        print("tessssssssssssssssssssssssssssssssst", time.time() - test_time)
        F1_sc.append(f1_rechner)
        print(f"f1 score für test dataset##{key}##", f1_rechner)
    Data_id.append("result")
    F1_sc.append(avr_result / len(X_test))
    result = {"data_id": Data_id, "f1_score": F1_sc}
    df = pd.DataFrame(result)
    df.to_excel("result_test2.xlsx")"""



    """Data_id=[]
    F1_sc=[]
    avr_result=0
    for key in X_test:
        Data_id.append(key)
        test_time=time.time()
        predictions = automl.predict(X_test[key])
        f1_rechner=sklearn.metrics.f1_score(y_test[key], predictions)
        avr_result=avr_result+f1_rechner
        print("tessssssssssssssssssssssssssssssssst",time.time()-test_time)
        F1_sc.append(f1_rechner)
        print(f"f1 score für test dataset##{key}##", f1_rechner)
    Data_id.append("result")
    F1_sc.append(avr_result/len(X_test))
    result= {"data_id": Data_id, "f1_score": F1_sc}
    df = pd.DataFrame(result)
    df.to_excel("result_test.xlsx")             ######anderung


    train_dataset = np.zeros((X_train.shape[0], 10))
    for index_feature in range(10):
        train_dataset[:, index_feature] = X_train[:, sort_idx[index_feature]].T
    print(train_dataset)
    print(type(train_dataset))
    print(train_dataset.shape)

    for c in sort_idx[:, 5]:
        print(c)
    plt.boxplot(r.importances[sort_idx].T,
                labels=[X_train[i] for i in sort_idx])

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    best_feature = np.zeros((X.shape[0], 3))
    for index_feature in range(3):
        best_feature[:, index_feature] = X[:, sort_idx[index_feature]]
    print(best_feature)
    print(best_feature.shape)


    for key in X_test:
        try:
            X = X_test[key]
            best_feature = np.zeros((X.shape[0], 10))
            for index_feature in range(10):
                best_feature[:, index_feature] = X[:, sort_idx[index_feature]].T
            print(best_feature)
            predictions = automl.predict(best_feature)
            print(f"f1 score für test dataset##{key}##", sklearn.metrics.f1_score(y_test[key], predictions))
        except:
            print("nööö")"""



    print("#######autosklearn phase:#####")
    """for dataset in X_test:
        for suchzeit in [30, 60, 120]:
            print(dataset)
            auto_model = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=suchzeit)
            start = time.time()
            auto_model.fit(X_test[dataset], y_test[dataset])
            runtime = time.time() - start
            pred = auto_model.predict(X_test[dataset])
            print("f11", sklearn.metrics.f1_score(pred, y_test[dataset]))
            print("zeit", suchzeit)
            print(runtime)"""



    """r = permutation_importance(automl, X_train, y_train, n_repeats=10, random_state=42)
    sort_idx = r.importances_mean.argsort()[::-1]

    train_dataset = np.zeros((X_train.shape[0], 10))
    for index_feature in range(10):
        train_dataset[:, index_feature] = X_train[:, sort_idx[index_feature]].T
    print(train_dataset)
    print(type(train_dataset))
    print(train_dataset.shape)"""
    #automl.fit(train_dataset, y_train)"""









    """for c in sort_idx[:, 5]:
        print(c)
    plt.boxplot(r.importances[sort_idx].T,
                labels=[X_train[i] for i in sort_idx])

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()"""
    #dataset_id = load_list("id_test_datasets.npy")
    #clf_list = load_list("classifier_list.npy")

    """best_feature = np.zeros((X.shape[0], 3))
    for index_feature in range(3):
        best_feature[:, index_feature] = X[:, sort_idx[index_feature]]
    print(best_feature)
    print(best_feature.shape)"""

    """for key in X_test:
        try:
            X= X_test[key]
            best_feature = np.zeros((X.shape[0], 10))
            for index_feature in range (10):
                best_feature[:, index_feature]= X[:, sort_idx[index_feature]].T
            predictions = automl.predict(best_feature)
            print(f"f1 score für test dataset##{key}##", sklearn.metrics.f1_score(y_test[key], predictions))
        except:
            print("nööö")"""








    """for dataset in X_test:
        try:
            print(f"autosklearn datasetid ###{dataset}####")
            baseline(X_test[dataset], y_test[dataset])
        except:
            print(f"datasetid###{dataset} contain Nan values")"""











    """print("#######autosklearn phase:#####")
    for suchzeit in [30, 60, 120]:
        for dataset in X_test:

            auto_model =autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=suchzeit)
            start = time.time()
            auto_model.fit(X_train, y_train)
            runtime = time.time() - start
            pred=auto_model.predict(X_test)
            print("f11",sklearn.metrics.f1_score(pred, y_test))
            print("zeit",suchzeit)
            print(runtime)"""

    """auto_model = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=suchzeit)
    start = time.time()
    auto_model.fit(X_train, y_train)
    runtime = time.time() - start
    pred = auto_model.predict(X_test)
    print("f11", sklearn.metrics.f1_score(pred, y_test))
    print("zeit", suchzeit)
    print(runtime)"""


    """print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)"""



    """
    X_big_test = np.load("X_big_test.npy", allow_pickle=True).item()
    y_big_test=np.load("y_big_test.npy", allow_pickle=True).item()
    
    for key in X_big_test:
        predictions = automl.predict(X_big_test[key])
        print(f"f1 score für test dataset##{key}##", sklearn.metrics.f1_score(y_big_test[key], predictions))"""
        #per_import(automl,X_big_test[key], y_big_test[key])


    """X_big_test, y_big_test = data_download(dataset_id, clf_list)
    for i in range (len(X_big_test)):
        predictions = automl.predict(X_big_test[i])
        print(f"f1 score für test dataset##{dataset_id[i]}##", sklearn.metrics.f1_score(y_big_test[i], predictions))"""

