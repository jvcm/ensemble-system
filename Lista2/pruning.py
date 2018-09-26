from sklearn.metrics import cohen_kappa_score
from itertools import combinations
import numpy as np
def kappa(pool = None, X_val = None, y_val = None):
    kappa = list()
    c = combinations(range(len(pool.estimators_)),2)
    tup = list()
    for i in c:
        tup.append(i)
        y1 = pool.estimators_[i[0]].predict(X_val)
        y2 = pool.estimators_[i[1]].predict(X_val)
        kappa.append(cohen_kappa_score(y1, y2))
    kappa = np.array(kappa)
    l = np.argsort(-kappa)
    estim = list()
    aux_sc = 0
    aux = pool.estimators_[:]
    for i in l:
        prune = list()
        if tup[i][0] not in estim:
            estim.append(tup[i][0])
        if tup[i][1] not in estim:
            estim.append(tup[i][1])        
        for c in estim:
            prune.append(aux[c])
        pool.estimators_ = prune
        score = pool.score(X_val, y_val)
        if score > aux_sc:
            aux_sc = score
            best = prune[:]
    return best    

def reduce_error(pool = None, X_val = None, y_val = None):
    estim = np.zeros(len(pool.estimators_))
    for i, est in enumerate(pool.estimators_):
        estim[i] = est.score(X_val, y_val)
    l = np.argsort(-estim)
    aux = pool.estimators_[:]
    best = list()
    best.append(pool.estimators_[l[0]])
    pool.estimators_ = best
    l = np.delete(l, 0)
    i = 1
    while l != np.array([]) :
        scores = np.zeros(len(l))
        score = pool.score(X_val, y_val)
        print(score)
        for k, j in enumerate(l):
            best.append(aux[j])
            pool.estimators_ = best
            scores[k] = pool.score(X_val, y_val)
            del best[i]
        print(scores)
        if score < scores.max():
            best_score_index = np.argsort(scores)[-1]
            best.append(aux[l[best_score_index]])
            l = np.delete(l, best_score_index)
            i += 1
        else:
            return best
