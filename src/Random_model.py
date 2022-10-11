from Clf_list import all_classifier
import random
def random_molde():
    all_clf = all_classifier()
    name, clf = random.choice(list(all_clf.items()))
    return name, clf