from collections import OrderedDict, defaultdict, deque
import scipy.sparse
import scipy.stats
from scipy.linalg import LinAlgError
from scipy.sparse import csr_matrix
import numpy as np
def class_entropy(X, y):
    labels = 1 if len(y.shape) == 1 else y.shape[1]

    entropies = []
    for i in range(labels):
        occurence_dict = defaultdict(float)
        for value in y if labels == 1 else y[:, i]:
            occurence_dict[value] += 1
        entropies.append(
            scipy.stats.entropy(
                [occurence_dict[key] for key in occurence_dict], base=2
            )
        )

    return np.mean(entropies)#


#muss überprüft werden
def symbols_sum(X,y):
    sum = np.nansum(X)
    return sum if np.isfinite(sum) else 0
from pymfe.mfe import MFE

a=MFE()
a.metafeature_description()
