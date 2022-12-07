import openml
import pandas as pd
from Meta_features_model import meta_features_model

import time

from Datasets_info import get_number_features, get_number_instance,f1_score_berechnen, get_balance, log_number_of_features,log_number_of_instances,class_occurences,class_probability_std, class_probability_mean, class_probability_max,class_probability_min, inverse_dataset_ratio,log_inverse_dataset_ratio, dataset_ratio, log_dataset_ratio,number_of_categorical_features,number_of_numeric_features,ratio_nominal_to_numerical,ratio_numerical_to_nominal,number_missing_values, percentage_mv, number_Features_mv,number_instances_mv,percentage_instances_mv,percentage_features_mv

from Numeric_transformer import transformer
from Datasets_info import get_number_features, get_number_instance
from Data_sampling import data_train_sampling, data_test_sampling
from to_unified import unified_data_test, unified_data_train
from scipy.sparse import csr_matrix
import numpy as np
from Clf_list import classfier, random_molde
from Helper import load_list,save_list
def download_data_train (data_id, clf_list):
    """"Download the OpenML datasets:(ID of dataset list)"""

    print(data_id)
    dataset = openml.datasets.get_dataset(dataset_id=data_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="array",
                                                                    target=dataset.default_target_attribute)

    preproceser = transformer(categorical_indicator)
    X_transformer = preproceser.fit_transform(X, y)
    X_unified, y_unified, column_name= unified_data_train(csr_matrix(X_transformer).toarray(), y, clf_list)

    X_unified = np.insert(X_unified, X_unified.shape[1], values=get_number_features(X), axis=1)
    column_name.append("nbr features")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=get_number_instance(X), axis=1)
    column_name.append("nbr instances")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=get_balance(y), axis=1)
    column_name.append("balance")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=log_number_of_features(X), axis=1)
    column_name.append("log nbr features")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=log_number_of_instances(X), axis=1)
    column_name.append("log nbr instances")
    cls_occ=class_occurences(X, y)
    X_unified = np.insert(X_unified, X_unified.shape[1], values=cls_occ[0], axis=1)
    column_name.append("0_occurences")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=cls_occ[1], axis=1)
    column_name.append("1_occurences")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=class_probability_std(X, y), axis=1)
    column_name.append("proba_std")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=class_probability_mean(X, y), axis=1)
    column_name.append("proba_mean")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=class_probability_max(X, y), axis=1)
    column_name.append("proba_max")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=class_probability_min(X, y), axis=1)
    column_name.append("proba_min")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=inverse_dataset_ratio(X), axis=1)
    column_name.append("inv ratio")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=log_inverse_dataset_ratio(X), axis=1)
    column_name.append("log invr atio")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=dataset_ratio(X), axis=1)
    column_name.append("ratio")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=log_dataset_ratio(X), axis=1)
    column_name.append("log ratio")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=number_of_categorical_features(categorical_indicator), axis=1)
    column_name.append("cat features")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=number_of_numeric_features(categorical_indicator), axis=1)
    column_name.append("numeric features")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=ratio_nominal_to_numerical(categorical_indicator), axis=1)
    column_name.append("nominal/numerical")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=ratio_numerical_to_nominal(categorical_indicator), axis=1)
    column_name.append("numerical/nominal")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=number_missing_values(X), axis=1)
    column_name.append("nbr MV")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=percentage_mv(X), axis=1)
    column_name.append("percent MV")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=number_Features_mv(X), axis=1)
    column_name.append("nbr features MV")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=number_instances_mv(X), axis=1)
    column_name.append("nbr instance MV")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=percentage_instances_mv(X), axis=1)
    column_name.append(" per inst MV")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=percentage_features_mv(X), axis=1)
    column_name.append("percent features MV")

    """except:
        print(f"datasetset=={data_id} nicht geeignet für unsere methode")"""

    """: type X_list, y_list: list """
    return np.array(X_unified), np.array(y_unified), column_name





def downlaod_data_test (data_id, clf_list):
    """"Download the OpenML datasets:(ID of dataset list)"""

    print(id)


    dataset = openml.datasets.get_dataset(dataset_id=data_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="array",
                                                                    target=dataset.default_target_attribute)
    preproceser = transformer(categorical_indicator)
    X_transformer = preproceser.fit_transform(X, y)
    X_unified, y_unified, column_name = unified_data_test(csr_matrix(X_transformer).toarray(), y, clf_list)

    X_unified = np.insert(X_unified, X_unified.shape[1], values=get_number_features(X), axis=1)
    column_name.append("nbr features")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=get_number_instance(X), axis=1)
    column_name.append("nbr instances")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=get_balance(y), axis=1)
    column_name.append("balance")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=log_number_of_features(X), axis=1)
    column_name.append("nbr features")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=log_number_of_instances(X), axis=1)
    column_name.append("log nbr instances")
    cls_occ=class_occurences(X, y)
    X_unified = np.insert(X_unified, X_unified.shape[1], values=cls_occ[0], axis=1)
    column_name.append("0_occurences")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=cls_occ[1], axis=1)
    column_name.append("1_occurences")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=class_probability_std(X, y), axis=1)
    column_name.append("proba_std")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=class_probability_mean(X, y), axis=1)
    column_name.append("proba_mean")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=class_probability_max(X, y), axis=1)
    column_name.append("proba_max")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=class_probability_min(X, y), axis=1)
    column_name.append("proba_min")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=inverse_dataset_ratio(X), axis=1)
    column_name.append("inverse ratio")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=log_inverse_dataset_ratio(X), axis=1)
    column_name.append("log inverse ratio")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=dataset_ratio(X), axis=1)
    column_name.append("ratio")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=log_dataset_ratio(X), axis=1)
    column_name.append("log ratio")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=number_of_categorical_features(categorical_indicator),
                          axis=1)
    column_name.append("cat features")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=number_of_numeric_features(categorical_indicator),
                          axis=1)
    column_name.append("num features")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=ratio_nominal_to_numerical(categorical_indicator),
                          axis=1)
    column_name.append("nominal/numerical")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=ratio_numerical_to_nominal(categorical_indicator),
                          axis=1)
    column_name.append("numerical/nominal")
    ###
    X_unified = np.insert(X_unified, X_unified.shape[1], values=number_missing_values(X) , axis=1)
    column_name.append("nbr MV")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=percentage_mv(X), axis=1)
    column_name.append("percent MV")
    X_unified = np.insert(X_unified, X_unified.shape[1], values= number_Features_mv(X), axis=1)
    column_name.append("nbr features MV")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=number_instances_mv(X), axis=1)
    column_name.append("nbr instance MV")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=percentage_instances_mv(X), axis=1)
    column_name.append(" per inst MV")
    X_unified = np.insert(X_unified, X_unified.shape[1], values=percentage_features_mv(X), axis=1)
    column_name.append("percent features MV")


    """: type X_list, y_list: list """
    return np.array(X_unified), np.array(y_unified), column_name


def main_unified_data_train():
    dataset_id=load_list("id_training_datasets.npy")
    del dataset_id[12]
    del dataset_id[29]
    file_clf = np.load("fixed_200_classifier.npy", allow_pickle=True)
    clf_list = file_clf.tolist()
    print(clf_list)


    data_representation=[]
    start = time.time()
    for id in range (len(dataset_id)):#(len(dataset_id))
        print(id)
        try:
            X_unified,y_unified, features_names =download_data_train(dataset_id[id], clf_list)
            features_names.append("target")
            data_representation.append(np.column_stack((X_unified, y_unified)))
        except:
            print(f"this dataset cann not be generatet{dataset_id[id]}")
    df=pd.DataFrame(np.vstack(data_representation), columns=features_names)
    print(f"time consoming to unified data representation{time.time()-start}")
    return df.to_excel("evaluation_200_clf.xlsx")
main_unified_data_train()


def main_unified_data_test():
    dataset_id=load_list("id_test_datasets.npy")
    del dataset_id[3]
    del dataset_id[7]
    del dataset_id[6]
    file_clf = np.load("fixed_200_classifier.npy", allow_pickle=True)
    clf_list = file_clf.tolist()



    X_test=dict()
    y_test=dict()
    data_representation=[]
    gesamt_time = time.time()

    for id in range (len(dataset_id)):  #len(dataset_id)
        try:
            print(id)

            time_jeder_dataset = time.time()
            X_unified,y_unified, features_names=downlaod_data_test(dataset_id[id], clf_list)
            data_representation.append(np.column_stack((X_unified, y_unified)))
            X_test[dataset_id[id]]=X_unified
            y_test[dataset_id[id]]=y_unified

            run_time=time.time()-time_jeder_dataset
            print(f"time für jeder dataset=={run_time}")
        except:
            print(f"{dataset_id[id]}this dataset cann not be generated")
    features_names.append("target")
    df1=pd.DataFrame(np.vstack(data_representation), columns=features_names)
    save_list(X_test, "evalX_200_clf.npy")
    save_list(y_test, "evaly_200_clf.npy")
    print(f"gesamt run time{time.time() - gesamt_time}")
    #return df1.to_excel("evaluation_test_200_clf.xlsx")
    return print("end..............")


main_unified_data_test()

"""x_y=[]
    for i in range(len(X_big_test)):
        combined_unified_data = np.column_stack((X_big_test[i], y_big_test[i]))
        x_y.append(combined_unified_data)"""



"""df = pd.DataFrame(np.vstack(x_y))
df.to_excel("unified_data_train.xlsx")"""





"""df_rou=pd.DataFrame(list)
df_rou.to_excel("rourou.xlsx")
print(type(X_big_test))
print(type(y_big_test))

print(len(X_big_test))
print(len(y_big_test))"""








#X_big_test,y_big_test=data_download(dataset_id, clf_list)
#read_dictionary = np.load("X_big_test.npy",allow_pickle='TRUE').item()
"""for i in read_dictionary.values():
    print(i)
print(type(read_dictionary))
print(read_dictionary)
print(type(read_dictionary[41161]))
read_y = np.load("y_big_test.npy",allow_pickle='TRUE').item()
print(type(read_y))
print(read_y)"""
"""dataset_id = load_list("id_training_datasets.npy")
clf_list = classfier()
data_representation = []
big_x_train=dict()
big_y_train=dict()

for id in range(len(dataset_id)):
    print(id)
    try:
        X_unified, y_unified = download_data_train(dataset_id[id], clf_list)
        big_x_train[dataset_id[id]]=X_unified
        big_y_train[dataset_id[id]]=y_unified

    except:
        print(f"c est pas la paine{dataset_id[id]}")



save_list(big_x_train,"big_x_train.npy")
save_list(big_y_train,"big_y_train.npy")"""





