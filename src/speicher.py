import openml
from Numeric_transformer import transformer
from Datasets_info import get_number_features, get_number_instance
from Data_sampling import data_train_sampling, data_test_sampling
from to_unified import unified_data
from scipy.sparse import csr_matrix
import numpy as np
from Helper import load_list,save_list


def unified_data_train (data_id, clf_list):
    """"Download the OpenML datasets:(ID of dataset list)"""
    X_list = []
    y_list = []

    for id in range(len(data_id)):

        try:
            dataset = openml.datasets.get_dataset(dataset_id=data_id[id])
            X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="array",
                                                                            target=dataset.default_target_attribute)
            number_features = get_number_features(X)
            number_instances = get_number_instance(X)
            preproceser = transformer(categorical_indicator)
            X_transformer = preproceser.fit_transform(X, y)
            X_sampling, y_sampling, X_rest, y_rest = data_train_sampling(X_transformer, y)
            X_unified, y_unified = unified_data(csr_matrix(X_transformer).toarray(), y_sampling, clf_list,
                                                number_features, number_instances)
            X_list.append(X_unified)
            y_list.append(y_unified)
        except:
            print(f"datasetset=={data_id[id]} nicht geeignet für unsere methode")

    """: type X_list, y_list: list """
    return X_list, y_list