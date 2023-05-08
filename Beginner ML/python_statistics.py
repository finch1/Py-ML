## https://realpython.com/python-statistics/

'''
Python statistics is a built in python library for descriptive statists
NumPy single or muldimentional arrays. ndarray. routines for statistical analysis
SciPy based on NumPy with additional functinality
Pandas based on NumPy. hadbles one-dim data with Series objects. two-dim data with DataFrame objects
Matplotlib data visualization
'''
import numpy as np 
import pandas as pd 
import math
import statistics as sts 
import scipy as sp 

# lists containing some arbitray data
x = [8.0, 1, 2.5, 4, 28.0]
x_nan = [8.0, 1, 2.5, math.nan, 4, 28.0]

print(math.isnan(x_nan[3]))
print(np.isnan(x_nan[3]))

# create one-dim np.ndarray and pd.Series objects that correspond to x and x_nan
y, y_nan = np.array(x), np.array(x_nan)
z, z_nan = pd.Series(x), pd.Series(x_nan)

# mean
mean_ = sts.mean(x)
print(mean_)
fmean_ = sts.mean(x) # faster alternative to mean. returns float
print(fmean_)

# with numpy 
mean_ = np.mean(x)
print(mean_)

# Above, mean() is a function, but you can use the corresponding method .mean() as well
print(y.mean())
print(z_nan.mean())

'''
The weighted mean is very handy when you need the mean of a dataset containing items 
that occur with given relative frequencies. For example, say that you have a set in which 
20% of all items are equal to 2, 50% of the items are equal to 4, and the remaining 30% of 
the items are equal to 8. You can calculate the mean of such a set like this
>>> 0.2 * 2 + 0.5 * 4 + 0.3 * 8
4.8
'''
x = [8.0, 1, 2.5, 4, 28.0]
w = [0.1, 0.2, 0.3, 0.25, 0.15]

wmean = sum(x_ * w_ for(x_, w_) in zip(x, w)) / sum(w)
print(wmean)

y, z = np.array(x), pd.Series(w) # doesn't matter pd or np
wmean = np.average(y, weights=z)
print(wmean)

# element-wise product
print((y * z).sum() / z.sum())

# The sample median is the middle element of a sorted dataset. 
''' 
You can compare the mean and median as one way to detect outliers and asymmetry in
your data.
'''
'''
If the number of elements is odd, then there’s a single middle value, 
so these functions behave just like median().

If the number of elements is even, then there are two middle values. 
In this case, median_low() returns the lower and median_high() the higher middle value.
'''

x = [1, 2.5, 4, 8.0, 28.0]

median_ = np.median(x) # or np.nanmedian
print(median_)
# ignore last element
median_ = sts.median(x[:-1])
print(median_)

'''
Mode is the value in the dataset that occurs the most
'''

u = [2, 3, 2, 8, 12]
print(sts.mode(u)) # returns single value
u_ = pd.Series(u) # note that each element has an index
print(u_.mode())
v = [12, 15, 12, 15, 21, 15, 12]
# print(sts.multimode(v)) # returns list

''' Measures of Variability '''

'''
Variance quantifies the spread of the data. 
It shows numerically how far the data points are from the mean.
'''

x = [-2.5, -1.5, 0.5, 0.7, 2.8] # small variance
y = [-5, -2.5, 0.5, 1.7, 5.4] # large variance

# sample variance
xvar_ = np.var(x, ddof =1)
print(xvar_)
yvar_ = pd.Series(y).var(ddof =1)
print(yvar_)

# population variance
xvar_ = np.var(x, ddof =0)
print(xvar_)

'''
Standard Deviation: a quantity by how much the members of a group differ frim the mean value for the group
'''
# population standard deviation ddof = 0
y = [-5, -2.5, 0.5, math.nan, 1.7, 5.4]
dev = np.nanstd(np.array(y), ddof =1)
# pandas skips nanby default.

'''
The sample skewness measures the asymmetry of a data sample.
'''
sk = pd.Series(x).skew()

# SUMMARY #
result = pd.Series(y).describe()
print(result)


''' Measures of Correlation Between Pairs of Data '''

x = list(range(-10, 11))
y = [0, 2, 2, 2, 2, 3, 3, 6, 7, 4, 7, 6, 6, 9, 4, 5, 5, 10, 11, 12, 14]
x_, y_ = np.array(x), np.array(y)

# Covariance is a measure that quantifies the strength and direction 
# of a relationship between  a pair of variables

cov_matrix = np.cov(x_, y_)
print(cov_matrix)
# covariance of x and x matrix[0 0]
print(cov_matrix[0, 0])
print(np.cov(x_))
# covariance of x and x matrix[1 1]
print(cov_matrix[1, 1])
# covariance of x and y matrix[0 1] or [1 0]
print(cov_matrix[0, 1])


# pearson correlation coefficient
corr_matrix = np.corrcoef(x_, y_)
print(corr_matrix)
# correlation coefficient of x and x matrix[0 0]
print(corr_matrix[0, 0])
# correlation coefficient of x and x matrix[1 1]
print(corr_matrix[1, 1])
# correlation coefficient of x and y matrix[0 1] or [1 0]. Same as linear regression
print(corr_matrix[0, 1])

# with Pandas
r = pd.Series(x_).corr(pd.Series(y_))
print(r)

'''
Working with two-dim arrays
'''
a = np.array([[1, 1, 1],
               [2, 3, 1],
               [4, 9, 2],
               [8, 27, 4],
               [16, 1, 1]])

print(np.mean(a))               
print(np.median(a))
print(a.var(ddof=1))
print(a.var(axis=0, ddof=1))

# AXES
print(a[0:2,1:2])

# axis=0 - whole column
print(np.mean(a,axis=0))
print(np.mean(a[:,2]))

# axis=1 - whole row
print(np.mean(a,axis=1))
print(np.mean(a[3]))

print(pd.Series(a[:,1]).describe())

## DataFrames
row_names = ['first','second','third','fourth','fifth']
col_names = ['A', 'B', 'C']
df = pd.DataFrame(a, index=row_names, columns=col_names)
print(df)
print(df.mean(axis=1))
print(df.var(axis=1))

# using column names in dataframes
print(df['B'])
print(df['B'].var())
print(df.std(axis=0, ddof=1))
print(df.describe())

# access one item
print(df.describe().at['mean', 'A'])

'''
df.values and df.to_numpy() give you a NumPy array with all items from the DataFrame without 
row and column labels. Note that df.to_numpy() is more flexible becaus
specify the data type of items and whether you want to use the existing data or copy it.
'''

import matplotlib.pyplot as plt 
plt.style.use('ggplot')

# Normally distributed numbers are generated with np.random.randn().
# Uniformly distributed integers are generated with np.random.randint().

'''
The parameters of .boxplot() define the following:
- x is your data.
- vert sets the plot orientation to horizontal when False. The default orientation is vertical.
- showmeans shows the mean of your data when True.
- meanline represents the mean as a line when True. The default representation is a point.
- labels: the labels of your data.
- patch_artist determines how to draw the graph.
- medianprops denotes the properties of the line representing the median.
- meanprops indicates the properties of the line or dot representing the mean.

The mean is the red dashed line.
The median is the purple line.
The first quartile is the left edge of the blue rectangle.
The third quartile is the right edge of the blue rectangle.
The interquartile range is the length of the blue rectangle.
The range contains everything from left to right.
The outliers are the dots to the left and right.
'''

np.random.seed(seed=0)
x = np.random.randn(1000) # number of items generated
y = np.random.randn(100) # number of items generated
z = np.random.randn(10) # number of items generated

fig, ax = plt.subplots()
ax.boxplot((x,y,z), vert=False, showmeans=True, meanline=True, labels=('x', 'y', 'z'), patch_artist=True,
            medianprops={'linewidth':2, 'color':'purple'},
            meanprops={'linewidth':2, 'color':'red'})

#plt.show()

# cumulative histogram adds the previous bins

hist, bin_edges = np.histogram(x, bins=10)
print(hist)
print(bin_edges)
fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=False)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
#plt.show()

fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=True)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
#plt.show()

fig, ax = plt.subplots()
ax.pie((len(x), len(y), len(z)), labels=('x', 'y', 'z'), autopct='%1.1f%%')
#plt.show()

from scipy.stats import linregress

x = np.arange(21) # 0 - 20
y = 5 + 2 * x + 2 * np.random.randn(21) # range distorted with some noise
slope, intercept, r, *__ = linregress(x, y)
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'

fig, ax = plt.subplots()
ax.plot(x, y, linewidth = 0, marker='s', label='Data points')
ax.plot(x, intercept+slope*x, label=line)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(facecolor='white')
plt.show()

# heat map



