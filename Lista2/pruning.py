import numpy as np

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
    score = 0
    while l != np.array([]) :
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
        else:
            print(len(best))
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
    pool.estimators_ = best
    l = np.delete(l, 0)
    score = pool.score(X_val, y_val)
    for j in l:
        best.append(aux[j])
        pool.estimators_ = best
        new_score = pool.score(X_val, y_val)
        if score < new_score:
            score = new_score
            aux = best[:]
    print(len(aux))
    return aux
