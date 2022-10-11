from Clf_list import random_molde
import pickle
import os
from Helper import save_list,load_list
os.system("clear")
def generate_clf_list():
    clf_list = []
    name_clf = []
    for model in range(50):
        name, clf = random_molde()
        clf_list.append(clf)
        name_clf.append(name)
    return clf_list, name_clf
"""clf_list, name_clf= generate_clf_list()

pickle.dump(clf_list,open("classifier_list.dat", "wb"))
save_list(name_clf, "classifier_name.npy")
cll= pickle.load(open("classifier_list.dat", "rb"))
name=load_list("classifier_name.npy")
print(name, type(name), len(name), name[4])"""








