from sklearn.neighbors import KNeighborsClassifier
import numpy as np
def kDN(X, y):
    knn = KNeighborsClassifier()
    knn.fit(X, y)
    hard_inst = np.zeros(len(y))
    for i, yi in enumerate(y[knn.kneighbors(X)[1]]):
        hard_inst[i] = list(yi == y[i]).count(False)/ knn.n_neighbors
    return hard_inst
