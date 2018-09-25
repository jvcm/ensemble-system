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

def best_first_pruning(pool = None, X_val = None, y_val = None, M = 0):
    estim = np.zeros(len(pool.estimators_))
    for i, est in enumerate(pool.estimators_):
        estim[i] = est.score(X_val, y_val)
    l = np.argsort(-estim)
    aux = pool.estimators_[:]
    best = list()
    best.append(pool.estimators_[l[0]])
    l = np.delete(l, 0)
    for i in range(1, M):
        scores = np.zeros(len(l))
        for k, j in enumerate(l):
            best.append(aux[j])
            pool.estimators_ = best
            scores[k] = pool.score(X_val, y_val)
            del best[i]
        best_score_index = np.argsort(scores)[-1]
        best.append(aux[l[best_score_index]])
        l = np.delete(l, best_score_index)
    return best
