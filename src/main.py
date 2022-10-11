import autosklearn.classification
from autosklearn.metrics import f1
from Load_datasets import load_unified_data
import time
import sklearn

import warnings
warnings.simplefilter(action='ignore')

from sklearn.metrics import accuracy_score



if __name__ == '__main__':
    start_time=time.time()
    #unified data to train our automl model
    X_train, y_train= load_unified_data("train_unified_data.xlsx")
    #unified data to test our automl model
    X_test, y_test = load_unified_data("test_unified_data.xlsx")
    automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=600, per_run_time_limit=10,
                                                             metric=f1)
    automl.fit(X_train, y_train)
    model_score=automl.score(X_train, y_train)
    print(f"automl model score {model_score}")
    predictions = automl.predict(X_test)
    print("f1 score für test dataset", sklearn.metrics.f1_score(y_test, predictions))
