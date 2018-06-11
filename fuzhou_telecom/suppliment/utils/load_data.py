from __future__ import absolute_import, division, print_function
import numpy as np
from sklearn.datasets import make_blobs

def load_salary():
	np.random.seed(1)
	x=np.random.randint(low=30, high=500,size=100)*100
	y=-np.round((x-500)*(x-90000)/100000 + np.array([(0.1 if s<15000 else 0.05)*s*np.random.randn() for s in x])) 
	x=x.reshape((-1,1))/10000.
	y=y/10000.
	return x, y

def load_blobs():
	seed = 125
	x, y = make_blobs(centers=2, random_state=seed)
	print("blob seed: %d" % seed)
	return x, y
	
def load_pca():
    np.random.seed(5)
    X_ = np.random.normal(size=(300, 2))
    X = np.dot(X_, np.random.normal(size=(2, 2))) + np.random.normal(size=2)
    return X