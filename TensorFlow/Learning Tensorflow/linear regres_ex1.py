# in this problem we are interested in retrieving a set of weights w
# and a bias b, assuming our target value is a linear combination of some 
# input vector x, with an additional Gaussian noise added to each sample

import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
import random as rd 

#create data and simulate results
x_data = np.random.randn(3, 2000)
w_real = [0.3, 0.5, 0.1]
b_real = -0.2

noise = np.random.randn(1, 2000)*0.1
y_data = np.matmul(w_real, x_data) + b_real + noise
print("X_DATA: ", x_data.shape)
print("NOISE: ", noise.shape)
print("Y_DATA: ", y_data.shape)
plt.subplot(221)
plt.imshow(x_data, interpolation='nearest')

y_min = min(y_data.T)
y_max = max(y_data.T)
print(y_min)
print(y_max)
y_min = np.around(y_min, decimals=2)
y_max = np.around(y_max, decimals=2)
print(y_min)
print(y_max)


x = np.arange(y_min, y_max, 0.0005)
xr = np.around(x, decimals=2)
print("XR: ", xr.shape)
plt.scatter(xr, noise.T, label = "NOISE", color='c')
plt.scatter(xr, y_data.T, label = "Y-DATA", color='m')
plt.legend()
plt.show()