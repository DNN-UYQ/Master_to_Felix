from sklearn.metrics import f1_score


def get_number_features(X):
    return X.shape[1]


def get_number_instance(X):
    return X.shape[0]


def f1_score_berechnen(X_rest, y_rest, clf):
    y_pred = clf.predict(X_rest)
    score = f1_score(y_rest, y_pred)

    return score
