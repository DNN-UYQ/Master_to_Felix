







import openml
from sklearn.pipeline import Pipeline
from sklearn import impute

from sklearn.preprocessing import StandardScaler
from Clf_list import classfier
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer


def transformer(categorical_indicator):
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

    my_transformers = []
    if np.sum(np.invert(categorical_indicator)) > 0:
        my_transformers.append(("num", numeric_transformer, np.invert(categorical_indicator)))
    categorical_transformer = OneHotEncoder()
    if np.sum(categorical_indicator) > 0:
        my_transformers.append(("cat", categorical_transformer, categorical_indicator))
    data_preprocessor = ColumnTransformer(my_transformers)

    return data_preprocessor










"""import openml
from sklearn.pipeline import Pipeline
from sklearn import impute

from sklearn.preprocessing import StandardScaler
from Clf_list import classfier
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer


def transformer(categorical_indicator):
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

    my_transformers = []

        if np.sum(np.invert(cat_inc)) > 0:
            my_transformers.append(("num", numeric_transformer, np.invert(cat_inc)))
        categorical_transformer = OneHotEncoder()
        if np.sum(cat_inc) > 0:
            my_transformers.append(("cat", categorical_transformer, cat_inc))
    data_preprocessor = ColumnTransformer(transformers=my_transformers)

    return data_preprocessor"""