# importing numpy, matplotlib and sklearn libraries
import matplotlib.pyplot as plt 
import numpy as np 

# importing datasets from scikit-learn
from sklearn import datasets, linear_model

# load the dataset
house_price = [245, 312, 279, 308, 199, 219, 405, 324, 319, 255]
size = [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]

# reshape the input to your regression
# -1 refers to as many columns as needed
size2 = np.array(size).reshape((-1,1))

# by using fit module in linear regression, user can fit the data
# frequently and quickly
## assign model to variable
regr = linear_model.LinearRegression()
## fit data into a linear regression model to see how close it 
## can get to the straight line
regr.fit(size2, house_price)
print("Coefficients: \n", regr.coef_)
print("Intercept: \n", regr.intercept_)

# formula obtained for the trained model
def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x, y)

# plotting the prediction line
graph('regr.coef_*x + regr.intercept_', range(1000, 2700))
plt.scatter(size, house_price, color='black')
plt.ylabel('house price')
plt.xlabel('size of house')
plt.show()