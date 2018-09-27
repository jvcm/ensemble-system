import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.base import clone

def reduce_error(pool = None, X_val = None, y_val = None):
    estim = np.zeros(len(pool.estimators_))
    for i, est in enumerate(pool.estimators_):
        estim[i] = est.score(X_val, y_val)
    l = np.argsort(-estim)
    aux = pool.estimators_[:]
    best = list()
    best.append(pool.estimators_[l[0]])
    l = np.delete(l, 0)
    i = 1
    score = 0
    
    while len(l) > 0 :
        scores = np.zeros(len(l))
        
        for k, j in enumerate(l):
            best.append(aux[j])
            pool.estimators_ = best
            scores[k] = pool.score(X_val, y_val)
            del best[i]
        if score < scores.max():
            best_score_index = np.argsort(scores)[-1]
            best.append(aux[l[best_score_index]])
            l = np.delete(l, best_score_index)
            i += 1
        elif score >= scores.max():
            print('*red',len(best))
            pool.estimators_ = aux
            return best
        score = pool.score(X_val, y_val)

def best_first(pool = None, X_val = None, y_val = None):
    estim = np.zeros(len(pool.estimators_))
    for i, est in enumerate(pool.estimators_):
        estim[i] = est.score(X_val, y_val)
    l = np.argsort(-estim)
    aux = pool.estimators_[:]
    best = list()
    best.append(pool.estimators_[l[0]])
    l = np.delete(l, 0)
    score = 0.0
    best_final = best[:]
    for j in l:
        best.append(aux[j])
        pool.estimators_ = best
        new_score = pool.score(X_val, y_val)
        if score < new_score:
            score = new_score
            best_final = best[:]
        elif score >= new_score: 
            continue
    print('*best',len(best_final))
    pool.estimators_ = aux
    return best_final
