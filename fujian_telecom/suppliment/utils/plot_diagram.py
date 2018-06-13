from __future__ import absolute_import, division, print_function
import numpy as np
from matplotlib import pyplot as plt

def plot_blobs(x, y):
	ax = plt.gca()
	ax.scatter(x[y == 0, 0], x[y == 0, 1], 
	            c='blue', s=40, label='0')
	ax.scatter(x[y == 1, 0], x[y == 1, 1], 
	            c='red', s=40, label='1', marker='s')
	ax.set_xlabel('first feature', fontsize=14)
	ax.set_ylabel('second feature', fontsize=14)
	ax.legend(loc='upper right');


def plot_2d_boundary(clf, xmin, xmax, ymin, ymax):
	x = np.linspace(xmin, xmax, 100)
	y = np.linspace(ymin, ymax, 100)
	xx, yy = np.meshgrid(x, y)
	grid_points = np.c_[xx.flatten(), yy.flatten()]
	try:
		levels = [0]
		distances = clf.decision_function(grid_points)
	except AttributeError:
		levels = [0.5]
		distances = clf.predict_proba(grid_points)[:,1]
	ax = plt.gca()
	ax.contour(xx, yy, distances.reshape(xx.shape), levels=levels)
