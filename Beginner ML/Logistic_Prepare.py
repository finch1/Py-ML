import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns 
from sklearn.model_selection import train_test_split

incoming = pd.read_csv('incoming.csv')
##print(list(incoming.columns))
# get an idea of the data set
# incoming.head()

# count how many empty fields per column
# incoming.isnull().sum()

# fill missing values 
incoming['CUSTOMER.COUNTRY'].fillna('XX', inplace=True)
incoming['overide'].fillna('clean', inplace=True)

# adding 
# incoming['new col'] = incoming['col 1'] + incoming['col 2']

# droping
# incoming.drop([col 1], axis=1, inplace=True) # axis 1 means whole column. inplace means do it on the data set

# aggregate
incoming.groupby(incoming['CREDIT.CUSTOMER'])['DEBIT.AMOUNT'].mean()
df = incoming.groupby(incoming['overide'])

# convert value to numeric
_overide = {'backdated' : 0, 'inactive' : 1, 'incorrect' : 2, 'locked' : 3, 'over limit' : 4, 'suspect' : 5, 'clean' : 6}
incoming['overide'] = incoming['overide'].map(_overide)

# drop unnecessary variables
# incoming.drop(['col1','col2'], axis=1, inplace=True) # drop columns not rows. alter dataset in place

# output to csv
# incoming.to_csv('../../data_output.csv', index=False) # do not output the first column index

# select features for prediction
features = incoming.drop('overide', axis=1)
labels = incoming['overide']

# this method can only split in two. specify the percentage split
# train , test + validation
# train data, test data, train labels, test labels
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.4, random_state=42) # inditialize randomizer
# test , validation
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state=42)

for dataset in [y_train, y_val, y_test]:
    print(round(len(dataset) / len(labels),2)) # percentage / full amount

# keep copy of same labels and features
X_train.to_csv('X_train.csv', index=False, header=True)
X_test.to_csv('X_test.csv', index=False, header=True)
y_train.to_csv('y_train.csv', index=False, header=True)
y_test.to_csv('y_test.csv', index=False, header=True)
X_val.to_csv('X_val.csv', index=False, header=True)
y_val.to_csv('y_val.csv', index=False, header=True)