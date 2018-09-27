from sklearn.metrics import cohen_kappa_score
from itertools import combinations
import numpy as np

def __coefficients(y1 = None, y2 = None, y_true = None):
    N11 = 0
    N10 = 0
    N01 = 0
    N00 = 0
    for i in range(len(y_true)):
        if (y1[i] == y_true[i] and y2[i] == y_true[i]):
            N11 += 1
        elif (y1[i] == y_true[i] and y2[i] != y_true[i]):
            N10 += 1
        elif(y1[i] != y_true[i] and y2[i] == y_true[i]):
            N01 += 1
        elif (y1[i] != y_true[i] and y2[i] != y_true[i]):
            N00 += 1
    coefs = np.array([N11, N10, N01, N00])
    return coefs

def kappa(pool_list = [], X_val = None, y_val = None):
    kappa = list()
    c = combinations(range(len(pool_list)),2)
    tup = list()
    for i in c:
        tup.append(i)
        y1 = pool_list[i[0]].predict(X_val)
        y2 = pool_list[i[1]].predict(X_val)
        kappa.append(cohen_kappa_score(y1, y2))
    kappa = np.array(kappa)
    return kappa.mean()

def disagreement(pool_list = [], X_val = None, y_val = None):
    disagree = list()
    c = combinations(range(len(pool_list)),2)
    tup = list()
    Sum_Qik = 0
    N = 0
    for i in c:
        disagree.append(i)
        y1 = pool_list[i[0]].predict(X_val)
        y2 = pool_list[i[1]].predict(X_val)
        d = __coefficients(y1, y2, y_val)
        Sum_Qik += (d[1] + d[2])/(d.sum())
        N += 1
    return Sum_Qik/N



