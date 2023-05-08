import numpy as np 
from random import randint
from sklearn.preprocessing import MinMaxScaler

train_labels = []
train_samples = []

for i in range(1000):
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    random_older = randint(65,100)
    train_samples.append(random_younger)
    train_labels.append(1)

for i in range(50):
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    random_older = randint(65,100)
    train_samples.append(random_younger)
    train_labels.append(0)

# Keras is expecting numpy array.     
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

# Make it easier for neural net to learn from and make better predictions
scaler = MinMaxScaler(feature_range=(0,1)) # scale ages between 0 and 1
scaler_train_samples = scaler.fit_transform((train_samples).reshape(-1,1)) # reshape -> fit-transform doesn't accept 1D array