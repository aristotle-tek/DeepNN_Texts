# Simple NN example to solve Table 1.

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

import math


# enter the matrix from Table 1
document_term_matrix = np.array([[1,0, 1, 0],[1,0,0,1],[0, 1,1,0],[0,1,0,1]], "float32")

y = np.array([[1],[0],[0],[1]], "float32")


model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu')) # Keras 'Dense' output = activation(dot(input, kernel) + bias
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(document_term_matrix, y, epochs=200)

print model.predict(document_term_matrix).round()



"""
model.summary() 49 parameters:
The first layer is (32 + 8) parameters
The second layer is (8 + 1) parameters

Now that we have the parameters, it is easy to see
that we can simply use the structure of the model
to predict the outcomes:
"""

M, b1, v, b2 = model.get_weights()

s1 = np.dot(document_term_matrix, M) + b1
s1 = np.maximum(s1, 0) # here is our rectified linear unit.

s2 = np.dot(s1, v) + b2

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

predictions = [round(sigmoid(x)) for x in s2]

print(predictions)

