import autosklearn.classification
from sklearn.model_selection import train_test_split
import time
import sklearn
from autosklearn.metrics import f1

import numpy as np




def baseline(X, y):



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=42, stratify=y)

    for suchzeit in [30, 60, 120]:
        print(f"suchezeit{suchzeit}")
        model_autoslearn = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=suchzeit,
                                                                  metric=f1)
        start_autosklearn = time.time()

        model_autoslearn.fit(X_train, y_train)
        run_time = time.time() - start_autosklearn
        predictions = model_autoslearn.predict(X_test)
        print("#####run time baseline###",run_time)
        print(f"####f1 score für test dataset##", sklearn.metrics.f1_score(y_test, predictions))
    return print("fin##fin")