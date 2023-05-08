# https://www.lynda.com/Google-TensorFlow-tutorials/Building-Deep-Learning-Applications-Keras-2-0/601801-2.html
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense

# Load training data set from CSV file
training_data_df = pd.read_csv("D:\\Ex_Files_Building_Deep_Learning_Apps\\03\\sales_data_training.csv")

# Load testing data set from CSV file
test_data_df = pd.read_csv("D:\\Ex_Files_Building_Deep_Learning_Apps\\03\\sales_data_test.csv")

# Data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well.
scaler = MinMaxScaler(feature_range=(0,1))

# Scale both the training inputs and outputs
# Fit transform, first fit scaler to our data, but figure out how much to scale down
# the numbers in each column and then we want it to actually transform or scale our data.
scaled_training = scaler.fit_transform(training_data_df)
# calling transform instead of fit transform means the scaler applies the same amount 
# of scaling to the test data as the training data. 
scaled_testing = scaler.transform(test_data_df)

# Print out the adjustment that the scaler applied to the total_earnings column of data
print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8], scaler.min_[8]))

# Create new pandas DataFrame objects from the scaled data
scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)

# Save scaled data dataframes to new CSV files
scaled_training_df.to_csv("D:\\Ex_Files_Building_Deep_Learning_Apps\\03\\sales_data_training_scaled.csv", index=False)
scaled_testing_df.to_csv("D:\\Ex_Files_Building_Deep_Learning_Apps\\03\\sales_data_test_scaled.csv", index=False)

training_data_df = pd.read_csv("D:\\Ex_Files_Building_Deep_Learning_Apps\\03\\sales_data_training_scaled.csv")

X = training_data_df.drop('total_earnings', axis=1).values
Y = training_data_df[['total_earnings']].values

# Define the model
# New Sequential model
model = Sequential()
# first layer - Dense(how many nodes, input size of nn (cols in csv), activation. )
model.add(Dense(50, input_dim = 9, activation = 'relu'))
model.add(Dense(100, activation = "relu"))
model.add(Dense(50, activation = "relu"))
# predicted earnings value should be a single linear value. So we use a linear activation fnction
model.add(Dense(1, activation="linear"))
model.compile(loss="mse", optimizer="adam")

# Train the model ( training features,
#                   expected outputs,
#                   training passes over training data during training process,
#                   shuffle order of training data,
#                   print more detailed info during training. )
model.fit(
    X,
    Y,
    epochs=7,
    shuffle=True,
    verbose=2
)

# Load the separate test data set for evaluation
test_data_df = pd.read_csv("D:\\Ex_Files_Building_Deep_Learning_Apps\\04\\sales_data_test_scaled.csv")

X_test = test_data_df.drop('total_earnings', axis=1).values
Y_test = test_data_df[['total_earnings']].values

test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

# save both the structure of the neural network and the trained weights that determine
# how the NN works. 
model.save("D:\\Ex_Files_Building_Deep_Learning_Apps\\04\\trained_model.h5")
print("Model saved to disk")
# Load the data we make to use to make a prediction
X = pd.read_csv("D:\\Ex_Files_Building_Deep_Learning_Apps\\04\\proposed_new_product.csv").values

# Make a prediction with the neural network
prediction = model.predict(X)

# Grab just the first element of the first prediction (since that's we only have one)
prediction = prediction[0][0]

# Re-scale the data from the 0-to-1 range back to dollars
# These constants are from when the data was originally scaled down to the 0-to-1 range
prediction = prediction + 0.1159
prediction = prediction / 0.0000036968

print("Earnings Prediction for Proposed Product - ${}".format(prediction))