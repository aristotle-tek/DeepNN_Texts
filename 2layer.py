"""
Under the hood - NN optimization by "hand",
 reflecting clear relationships with derivations 
 for partial derivatives. (Not optimized!)

for network with:
	layer1 = dense layer with ReLU activation function
	layer2 = dense layer sigmoid output

The first layer is (32 + 8) parameters
The second layer is (8 + 1) parameters

"""

import numpy as np
import copy
import math

from approx_deriv import *

np.random.seed(42)


def sigmoid(x):
	return 1 / (1 + math.exp(-x))


def mean_square_error(yhat, y):
	error = yhat-y
	return(error**2)


def forward_pass_1ex(x, M, b1, v, b2):
	s1 = np.dot(x, M) + b1
	s1 = np.maximum(s1, 0) # here is our rectified linear unit.
	s2 = np.dot(s1, v) + b2
	return(sigmoid(s2))


def forward_pass(x, M, b1, v, b2):
	if x.shape == (4,):
		return(forward_pass_1ex(x, M, b1, v, b2) )
	else:
		out = []
		for xi in range(x.shape[1]):
			xcurr = forward_pass_1ex(x[xi], M, b1, v, b2)
			out.append(xcurr)
		return(np.array(out))


def calc_partial_b2(x, y, M, b1, v, b2):
	forwardpass = forward_pass_1ex(x, M, b1, v, b2)
	term1 = 2*(forwardpass - y)
	term2 = (1-forwardpass)*forwardpass
	return term1*term2


def update_b2(x, y, M, b1, v, b2, learning_rate):
	partial_b2 = calc_partial_b2(x, y, M, b1, v, b2)
	b2 = b2 - learning_rate * partial_b2
	return b2


def calc_partial_v(x, y, M, b1, v, b2, whichcalc):
	""" 
	NB: for just 1 example x, not the whole batch X.
	"""
	inner = np.dot(x, M) + b1
	s1 = np.maximum(inner, 0)
	s2 = np.dot(s1, v) + b2
	forwardpass = sigmoid(s2)
	term1 = 2*(forwardpass - y)
	term2 = forwardpass * (1-forwardpass) * s1[whichcalc]
	return term1*term2


def update_all_v(x, y, M, b1, v, b2, learning_rate):
	new_vs = []
	for vi in range(len(v)):
		partial = calc_partial_v(x, y, M, b1, v, b2, vi)
		new_v = v[vi] - learning_rate * partial
		new_vs.append(new_v)
	return np.array(new_vs).squeeze()


def calc_partial_inner_layer(x, y, M, b1, v, b2, layer2_deltas, i, j, bias):
	""" i is only specified for a non-bias term, otherwise it is ignored"""
	rel_m = M[:,j]
	rel = np.dot(rel_m,x) + b1[j]
	if rel <= 0:
		return 0
	else:
		sumlayers = layer2_deltas * v[j]
		if bias:
			partial = 1 * sumlayers * 1
		else: 
			partial = 1 * sumlayers * x[i] 
		return partial


def update_all_b1(x, y, M, b1, v, b2, layer2_deltas, learning_rate):
	new_bs = []
	for bj in range(len(b1)):
		partial = calc_partial_inner_layer(x, y, M, b1, v, b2, layer2_deltas, 0, bj, True)
		new_b1 = b1[bj] - learning_rate * partial
		new_bs.append(new_b1)
	return np.array(new_bs).squeeze()


def calc_layer2_deltas(x, y, M, b1, v, b2):
	""" In this case just one number."""
	inner = np.dot(x, M) + b1
	s1 = np.maximum(inner, 0) # (applied elementwise)
	s2 = np.dot(s1, v) + b2
	forwardpass = sigmoid(s2)
	layer2delta = 2*(forwardpass - y) *forwardpass * (1-forwardpass) # in this case they are identical, don't need to calc each unique value.
	return layer2delta[0]


def update_all_M(x, y, M, b1, v, b2, layer2_deltas, learning_rate=.001):
	it = np.nditer(M, flags=['multi_index'])
	while not it.finished:
		i,j = it.multi_index
		partial = calc_partial_inner_layer(x, y, M, b1, v, b2, layer2_deltas, i, j, False)
		M[i,j] = M[i,j] - learning_rate * partial
		it.iternext()
	return np.array(M)




class twolayerNN(object):

	def __init__(self):
		self.M = abs(np.random.randn(4,8)*.1)
		self.b1 = abs(np.random.normal(scale=.1, size=8))
		self.v = abs(np.random.normal(scale=.1, size=8))
		self.b2 = abs(np.random.normal(scale=.1))

		self.X = np.array([[1,0, 1, 0],[1,0,0,1],[0, 1,1,0],[0,1,0,1]], "float32")
		self.Y = np.array([[1],[0],[0],[1]], "float32")

	def predict(self):
		yhat = forward_pass(self.X, self.M, self.b1, self.v, self.b2)
		print("Current prediction: %s" % (np.array(yhat).round() ) )


	def bp_update(self, x, y):
		""" update all params from a single example x, y"""
		M = self.M
		b1 = self.b1
		v = self.v
		b2 = self.b2
		layer2_deltas = calc_layer2_deltas(x, y, M, b1, v, b2)
		b2new = update_b2(x, y, M, b1, v, b2, .01)
		b1new = update_all_b1(x, y, M, b1, v, b2, layer2_deltas, .1)
		Mnew = update_all_M(x, y, M, b1, v, b2, layer2_deltas, .1)
		vnew = update_all_v(x, y, M, b1, v, b2, .01)
		self.b2, self.b1, self.M, self.v = b2new, b1new, Mnew, vnew
	
	def approx_update(self, x, y):
		M = self.M
		b1 = self.b1
		v = self.v
		b2 = self.b2
		b2new = approx_update_b2(x, y, M, b1, v, b2, 1e5)
		b1new = approx_update_all_b1(x, y, M, b1, v, b2, 1e5)
		Mnew = approx_update_all_M(x, y, M, b1, v, b2, 1e5)
		vnew = approx_update_all_v(x, y, M, b1, v, b2, 1e5)
		self.b2, self.b1, self.M, self.v = b2new, b1new, Mnew, vnew

	def train(self, epochs=1000, approx=False):
		for epoch in range(epochs):
			for example in range(4):
				x = self.X[example]
				y = self.Y[example]
				if approx:
					self.approx_update(x, y)
				else:
					self.bp_update(x, y)
			if epoch%200==0:
				print("b2=%.4f, b13=%.4f, v3=%.4f, m00=%.4f" % (self.b2, self.b1[3], self.v[3], self.M[0,0] ))
				self.predict()
		self.predict()


if __name__ == "__main__":
	print("\nIllustration of two-layer NN\n")
	nn = twolayerNN()
	nn.predict()
	nn.train(epochs=1000, approx=False)
	#nn.predict()

	print("\n\nApproximating gradiant using finite differences method:\n")
	nn2 = twolayerNN()
	nn2.predict()
	nn2.train(epochs=1000, approx=True)
	