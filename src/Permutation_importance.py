from sklearn.inspection import plot_partial_dependence, permutation_importance
import matplotlib.pyplot as plt


def per_import(automl, X_test, y_test):
    r = permutation_importance(automl, X_test, y_test, n_repeats=10, random_state=0)
    sort_idx = r.importances_mean.argsort()[::-1]

    plt.boxplot(r.importances[sort_idx].T,
                labels=[X_test[i] for i in sort_idx])

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    return plt.show()


