import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.insert(0, '../Lista2')
import kdn, pruning

class IHClassifier:
	def __init__(self):
		return

	def fit(self, X_train, y_train): 
		easy0 = []
		easy1 = []
		hard0 = []
		hard1 = []
		vis = kdn.kDN(X_train, y_train)
	    
		for i, k in enumerate(vis):
			if k < 0.4:
			    if y_train[i] == False:
			        easy0.append(i)
			    elif y_train[i] == True:
			        easy1.append(i)
			else:
			    if y_train[i] == False:
			        hard0.append(i)
			    elif y_train[i] == True:
			        hard1.append(i)
		print('Easy-False:',len(easy0),'Easy-True:',len(easy1),'Hard-False:',len(hard0),'Hard-True:',len(hard1))
		self.centroid_easy0 = X_train[easy0].mean(axis=0)
		self.centroid_easy1 = X_train[easy1].mean(axis=0)
		self.centroid_hard0 = X_train[hard0].mean(axis=0)
		self.centroid_hard1 = X_train[hard1].mean(axis=0)
		self.knn = KNeighborsClassifier(n_neighbors=3)
		self.knn.fit(X_train, y_train)
		return

	def predict(self, X_test):   
		predict = []   
		
		for i, xq in enumerate(X_test):
		    dist_cent = {}    
		    dist_cent[0] = np.linalg.norm(xq-self.centroid_easy0)
		    dist_cent[1] = np.linalg.norm(xq-self.centroid_easy1)
		    dist_cent[2] = np.linalg.norm(xq-self.centroid_hard0)
		    dist_cent[3] = np.linalg.norm(xq-self.centroid_hard1)
		    result = sorted(dist_cent, key = dist_cent.get)
		    
		    if ((result == 0) or (result == 1)):
		        predict.append(bool(result))
		    else:
		        xq = xq.reshape(1, -1)
		        predict.append(self.knn.predict(xq)[0])
		    
		predict = np.array(predict)
		return predict
