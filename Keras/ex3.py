# From the course: Neural Networks and Convolutional Neural Networks Essential Training
# FULLY CONNECTED NEURAL NETWORK

from keras.datasets import mnist
from keras.preprocessing.image import load_img, array_to_img
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

import numpy as np 
import matplotlib.pyplot as plt 
#%matplotlib inline # allows to view the plots in jypter notebooks

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# UNDERSTANDING THE IMAGE DATA FORMAT
'''
print(X_train.shape)
print(X_test.shape)
print(y_test.shape)
print(X_train[0].shape)
plt.imshow(X_train[0], cmap='gray')
plt.show()
print(y_train[0])
'''
# PREPROCESSING THE IMAGE DATA
# get the data size
Train_layers, image_height, image_width = X_train.shape
Test_layers, image_height, image_width = X_test.shape

# will give us one layer with 784 neurons across
X_train = X_train.reshape(Train_layers, image_height*image_width)
X_test = X_test.reshape(Test_layers, image_height*image_width)
'''
print(X_train.shape)
print(X_test.shape) 
'''
# Lets change the variable type so we can divide by 255 to scale the data between 0 - 1.
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# scale data
X_train /= 255.0
X_test /= 255.0
'''
print(X_test[0])
'''
# categorize the labels from one class to 10 classes
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)
'''
print(y_test.shape)
'''

# BUILD MODEL
model = Sequential()
# our first layer with 512 neurons - so 512 outputs
model.add(Dense(512, activation='relu', input_shape=(image_height*image_width,)))
model.add(Dense(512, activation='relu'))
# soft max as we want one of the 10 classes
model.add(Dense(10, activation='softmax'))

# COMPILE THE MODEL
# before we train we need to compile our model
# categorical_crossentropy allows for 10 classes
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Parameter explenation
# input = 784 * 512 neurons + 512 biases = 401920
# input = 512 * 512 neurons + 512 biases = 262656
# input = 512 * 10 neurons + 10 biases = 5130

# TRAIN THE MODEL
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# What is the accuracy of the model?
# Plot the accuracy of the training model
plt.plot(history.history['acc'])
plt.show()
# Plot the accuracy of the training validation set
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.show()
# Plot the accuracy of the training set, validaion set and accuracy of the model
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.show()

# EVALUATING / TESTING THE MODEL
# result is a list
score = model.evaluate(X_test, y_test)
print(score)
