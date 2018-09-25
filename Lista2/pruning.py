from sklearn.metrics import cohen_kappa_score
from itertools import combinations
import numpy as np
def kappa_pruning(pool = None, validation = None, best_values = 0):
    kappa = list()
    c = combinations(range(len(pool.estimators_)),2)
    tup = list()
    for i in c:
        tup.append(i)
        y1 = pool.estimators_[i[0]].predict(validation)
        y2 = pool.estimators_[i[1]].predict(validation)
        kappa.append(cohen_kappa_score(y1, y2))
    kappa = np.array(kappa)
    l = np.argsort(-kappa)

    estim = list()
    for i in l[:best_values]:
        if tup[i][0] not in estim:
            estim.append(tup[i][0])
        if tup[i][1] not in estim:
            estim.append(tup[i][1])
            
    return estim
