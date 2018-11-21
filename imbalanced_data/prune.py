import numpy as np
from sklearn.ensemble import BaggingClassifier
from imblearn.metrics import geometric_mean_score
import pandas as pd

def reduce_error_GM(pool = None, X_val = None, y_val = None):
	estim = np.zeros(len(pool.estimators_))
	for i, est in enumerate(pool.estimators_):
		y_pred = est.predict(X_val)
		estim[i] = geometric_mean_score(y_val, y_pred)
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
			y_pred = pool.predict(X_val)
			scores[k] = geometric_mean_score(y_val, y_pred)
			del best[i]
		if score < scores.max():
			best_score_index = np.argsort(scores)[-1]
			best.append(aux[l[best_score_index]])
			l = np.delete(l, best_score_index)
			i += 1
		elif score >= scores.max():
			pool.estimators_ = aux
			return best
		pred = pool.predict(X_val)
		score = geometric_mean_score(y_val, pred)

def complementarity(pool = None, X_val = None, y_val = None):
	estim = np.zeros(len(pool.estimators_))
	for i, est in enumerate(pool.estimators_):
		y_pred = est.predict(X_val)
		estim[i] = geometric_mean_score(y_val, y_pred)
	l = np.argsort(-estim)
	aux = pool.estimators_[:]
	best = list()
	best.append(pool.estimators_[l[0]])
	l = np.delete(l, 0)
	