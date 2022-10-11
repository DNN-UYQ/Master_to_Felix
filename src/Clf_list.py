import random

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB


def classfier():
    clf1 = LogisticRegression(random_state=0)
    clf2 = KNeighborsClassifier(n_neighbors=3)
    clf3 = svm.SVC(gamma=0.001, probability=True)
    return [clf1, clf2, clf3]

def all_classifier():
    cl1 = AdaBoostClassifier(n_estimators=random.randint(1, 50),learning_rate=random.uniform(0.1, 1.0), algorithm=random.choice(["SAMME", "SAMME.R"]))
    cl2 = BernoulliNB(alpha=random.uniform(0.01, 10.0))
    cl3 = DecisionTreeClassifier(criterion=random.choice(["gini", "entropy"]), splitter=random.choice(["best", "random"]), min_samples_leaf=random.randint(2, 5), min_samples_split=random.randint(2, 4))
    cl4 = ExtraTreesClassifier(criterion=random.choice(["gini", "entropy"]), min_samples_leaf=random.randint(2, 5), min_samples_split=random.randint(2, 4))
    cl5 = GaussianNB(var_smoothing=random.uniform(1e-2, 1e-15))
    cl6 = HistGradientBoostingClassifier(learning_rate=random.random())
    cl7 = KNeighborsClassifier(n_neighbors=random.randint(1, 10), weights=random.choice(["uniform", "distance"]))
    cl8 = LinearDiscriminantAnalysis(solver=random.choice(["lsqr", "eigen"]), shrinkage=random.random())
    #cl9 = LinearSVC(probability=True)
    #cl10 = MLPClassifier(activation=random.choice(["identity", "logistic", "tanh", "relu"]), solver=random.choice(["lbfgs", "sgd", "adam"]), alpha=random.random(),learning_rate=random.choice(["constant", "invscaling", "adaptive"]))
    #cl11 = MultinomialNB(alpha=random.random(), fit_prior=random.choice(["True", "False"]))
    #cl12 = PassiveAggressiveClassifier(max_iter= random.randint(800, 1500), fit_intercept=random.choice(["True", "False"]))
    cl13 = QuadraticDiscriminantAnalysis(store_covariance=random.choice(["True", "False"]), tol=random.uniform(0.0001, 1.0), reg_param=random.uniform(0.0001, 1.0))
    cl14 = RandomForestClassifier(n_estimators=random.randint(50, 600), criterion=random.choice(["gini", "entropy"]), min_samples_leaf=random.randint(2, 5), min_samples_split=random.randint(2, 4))
    cl15 = SGDClassifier(penalty=random.choice(["l2", "l1", "elasticnet"]))

    cl16 = SVC(kernel=random.choice(["linear", "poly", "rbf", "sigmoid", "precomputed"]), probability=True)
    cl17 = LogisticRegression(penalty=random.choice(["l2"]), C=random.random())

    all_clf = {
        "AdaBoostClassifier": cl1,
        "BernoulliNB": cl2,
        "DecisionTreeClassifier": cl3,
        "ExtraTreesClassifier": cl4,
        "GaussianNB": cl5,
        "HistGradientBoostingClassifier": cl6,
        "KNeighborsClassifier": cl7,
        "LinearDiscriminantAnalysis": cl8,
        #"LinearSVC": cl9,
        #"MLPClassifier": cl10,
        #"MultinomialNB": cl11,
        #"PassiveAggressiveClassifier": cl12,
        "QuadraticDiscriminantAnalysis": cl13,
        "RandomForestClassifier": cl14,
        #"SGDClassifier": cl15,
        #"SVC": cl16,
        "LogisticRegression":cl17
    }
    return all_clf
#It is a function, that should randomly generate classifier
def random_molde():
    all_clf = all_classifier()
    name, clf = random.choice(list(all_clf.items()))
    return name, clf

name, clf=random_molde()
print(clf)