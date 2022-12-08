import time
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
import openml
import autosklearn.classification
import pandas as pd
from autosklearn.metrics import f1
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split

dataset_id=np.load("id_test_datasets.npy", allow_pickle=True).tolist()


print (dataset_id)
dataset_id.remove(1111)
dataset_id.remove(41150)
dataset_id.remove(41147)
print (dataset_id)



"""list_id=[]
list_30=[]
list_60=[]
list_120=[]
sum_30=0
sum_60=0
sum_120=0
for dataset in dataset_id:
    list_id.append(dataset)
    print(dataset)
    test_dataset = openml.datasets.get_dataset(dataset_id=dataset)
    X, y, categorical_indicator, attribute_names = test_dataset.get_data(dataset_format="array",
                                                                         target=test_dataset.default_target_attribute)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=42, stratify=y)
    time_jederdataset=time.time()
    for suchzeit in [30, 60, 120]:
        auto_model = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=suchzeit, metric=f1)
        auto_model.fit(X_train,y_train)
        pred = auto_model.predict(X_test)
        ff=sklearn.metrics.f1_score(pred, y_test)
        if suchzeit==30:
            list_30.append(ff)
            sum_30=sum_30+ff
        elif suchzeit==60:
            list_60.append(ff)
            sum_60=sum_60+ff

        elif suchzeit==120:
            list_120.append(ff)
            sum_120=sum_120+ff

        print("f11 auto sklearn model testing", ff)
        print("zeit", suchzeit)
    print("time consoming",time.time()-time_jederdataset)
list_30.append(sum_30/len(list_id))
list_60.append(sum_60/len(list_id))
list_120.append(sum_120/len(list_id))
list_id.append("average")


create_dic= {"d.id":list_id, "S.Z30":list_30, "S.Z60":list_60,"S.Z120":list_120}
df = pd.DataFrame(create_dic)
df.to_excel("Baseline_Autosklearn.xlsx")"""