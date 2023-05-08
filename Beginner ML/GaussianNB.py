import numpy as np 
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt 
# supervised classification
X = np.array([[-1, -1],[-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]) # features
Y = np.array([1, 1, 1, 2, 2, 2]) # labels
clf = GaussianNB() # create classifier
clf.fit(X, Y) # training data & learn the patterns
GaussianNB()
# prediction : what do you think label is for this point
x_new = -0.8
y_new = -1
l_new = clf.predict([[x_new, y_new]])

x, y = X.T
plt.scatter(x, y, color="magenta")
for i, label in enumerate(Y):
    plt.annotate(label, (x[i],y[i]))

plt.scatter(x_new, y_new, color="cyan")
plt.annotate(l_new[0], (x_new, y_new))

plt.show()
