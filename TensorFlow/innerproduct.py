# -*- coding: utf-8 -*-
"""innerproduct.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-hqnrw0dEbskFR-8FpMbfehwfNOoyKO3
"""

import numpy as np
import random
import tensorflow as tf
import keras

# Create a neural net model to calculate the dot (inner) product 
# of two vectors (which are numpy int arrays) of size 2 
# and values that can either be 1 or 2.
# These limitations are to make the training reasonably fast.

def vector_dot_scratch(a,b):

  # The value is -1 if vectors have different shapes (lengths).
  # This is just for completeness here as model fitting does not support 
  # vectors of different shapes (ragged tensors).
  # See: https://github.com/tensorflow/tensorflow/issues/44988
  if(np.shape(a) != np.shape(b)):
     c = -1
  else:
    c = 0
    for i in range(np.array(a).size):
      c += a[i-1]*b[i-1]
  return c

# Sets lower and upper bounds of random integer values for the vector elements.
lower_rand_int = 1
upper_rand_int = 2

vector_length = 2

num_train_data = 7000
num_test_data = 3000

train_data = [0]*num_train_data
train_target = [0]*num_train_data
for i in range(num_train_data):
  a = [random.randint(lower_rand_int,upper_rand_int) for x in range(vector_length)]
  b = [random.randint(lower_rand_int,upper_rand_int) for x in range(vector_length)]
  train_data[i] = [a, b]
  train_target[i] = vector_dot_scratch(a,b)
 

test_data = [0]*num_test_data
test_target = [0]*num_test_data
for i in range(num_test_data):
 a = [random.randint(lower_rand_int,upper_rand_int) for x in range(vector_length)]
 b = [random.randint(lower_rand_int,upper_rand_int) for x in range(vector_length)]
 test_data[i] = [a, b]
 test_target[i] = vector_dot_scratch(a,b)


model = keras.Sequential([
                          
    # Flatten takes the two input vectors and combines them into one input
    # as sequential models can only take one input.
    # input_shape = (number of possible element values, vector_length)
    keras.layers.Flatten(input_shape=(2,vector_length)),
    keras.layers.Dense(2, activation='sigmoid'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', 
              loss='mse', # Mean Squared Error
              metrics=['mae']) # Mean Absolute Error

model.fit(train_data, train_target, epochs=50, batch_size=1)

test_loss, test_acc = model.evaluate(test_data, test_target)
print('Test accuracy:', test_acc)

# Prediction for inner product of [2,2] and [2,2] = 8.
a= np.array([[[2,2],[2,2]]])
print(model.predict(a))