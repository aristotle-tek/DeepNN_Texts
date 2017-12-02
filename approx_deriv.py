"""

Finite differences method to approximate the gradient to the error fn at (x,y)
partial L/ partial w = L(w+ epsilon) - L(w- epsilon) / 2epsilon

"""


import numpy as np
import math
import copy


#-------------------------------------------------------

#-------------------------------------------------------
def sigmoid(x):
	return 1 / (1 + math.exp(-x))


def initialize_weights():
	"""Randomly initialize weights from normal distrib
	better, with ReLU, to initialize to positive values
	Could also use truncated random.
	"""
	M = abs(np.random.randn(4,8)*.1)
	b1 = abs(np.random.normal(scale=.1, size=8))
	v = abs(np.random.normal(scale=.1, size=8))
	b2 = abs(np.random.normal(scale=.1))
	return(M, b1, v, b2)


def forward_pass(X, M, b1, v, b2):
	s1 = np.dot(X, M) + b1
	s1 = np.maximum(s1, 0) # here is our rectified linear unit.

	s2 = np.dot(s1, v) + b2
	if s2.shape == ():
		return(sigmoid(s2))
	else:
		return([sigmoid(x) for x in s2])


def calc_error(x, y, M, b1, v, b2):
	yhat = forward_pass(x, M, b1, v, b2)
	err = (yhat - y)**2
	return err


def approx_update_b2(x, y, M, b1, v, b2, learning_rate=1e5, epsilon=.001):
	approx = (calc_error(x, y, M, b1, v, b2+epsilon) - calc_error(x, y, M, b1, v, b2-epsilon)) /2*epsilon
	new_b2 = b2 - learning_rate * approx
	return(new_b2)


def calc_approx_partial_b1(x, y, M, b1, v, b2, epsilon, whichcalc):
	""" 
	NB: for just 1 example x, not the whole batch X.
	"""
	b1plus = copy.deepcopy(b1)
	b1min = copy.deepcopy(b1)
	rel = b1[whichcalc]
	b1plus[whichcalc] = rel+ epsilon
	b1min[whichcalc] = rel- epsilon
	partial = (calc_error(x, y, M, b1plus, v, b2) - calc_error(x, y, M, b1min, v, b2)) /2*epsilon
	return partial


def approx_update_all_b1(x, y, M, b1, v, b2, learning_rate=1e5, epsilon=.001):
	new_b1s = []
	for b1i in range(len(b1)):
		approx = calc_approx_partial_b1(x, y, M, b1, v, b2, epsilon, b1i)
		new_b1 = b1[b1i] - learning_rate * approx
		new_b1s.append(new_b1)
	return np.array(new_b1s).squeeze()


def calc_approx_partial_v(x, y, M, b1, v, b2, epsilon, whichcalc):
	""" 
	NB: for just 1 example x, not the whole batch X.
	"""
	vplus = copy.deepcopy(v)
	vmin = copy.deepcopy(v)
	rel = v[whichcalc]
	vplus[whichcalc] = rel+ epsilon
	vmin[whichcalc] = rel- epsilon
	partial = (calc_error(x, y, M, b1, vplus, b2) - calc_error(x, y, M, b1, vmin, b2)) /2*epsilon
	return partial


def approx_update_all_v(x, y, M, b1, v, b2, learning_rate=1e5, epsilon=.001):
	new_vs = []
	for vi in range(len(v)):
		approx = calc_approx_partial_v(x, y, M, b1, v, b2, epsilon, vi)
		new_v = v[vi] - learning_rate * approx
		new_vs.append(new_v)
	return np.array(new_vs).squeeze()


def calc_approx_partial_M(x, y, M, b1, v, b2, epsilon, i, j):
	""" 
	NB: for just 1 example x, not the whole batch X.
	"""
	Mplus = copy.deepcopy(M)
	Mmin = copy.deepcopy(M)
	rel = M[i, j]
	Mplus[i,j] = rel+ epsilon
	Mmin[i,j] = rel- epsilon
	partial = (calc_error(x, y, Mplus, b1, v, b2) - calc_error(x, y, Mmin, b1, v, b2)) /2*epsilon
	return partial

def approx_update_all_M(x, y, M, b1, v, b2, learning_rate=10e5, epsilon=.001):
	it = np.nditer(M, flags=['multi_index'])
	while not it.finished:
		i,j = it.multi_index
		approx = calc_approx_partial_M(x, y, M, b1, v, b2, epsilon, i, j)
		M[i,j] = M[i,j] - learning_rate * approx
		it.iternext()
	return np.array(M)




