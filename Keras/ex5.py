# From the course: Building Deep Learning Applications with Keras 2.0
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import *

# load training data set from CSV file
training_data_df = pd.read_csv("Ex_Files_Building_Deep_Learning_Apps/03/sales_data_training.csv")

# Load testing data set from CSV file
testing_data_df = pd.read_csv("Ex_Files_Building_Deep_Learning_Apps/03/sales_data_test.csv")

# Data needs to be scaled to a small range like 0 to 1 for the NN to work well
scaler = MinMaxScaler(feature_range=(0,1))

# Scale both the training inputs and outputs
# fit_transform figures out how to scale the data
scaled_training = scaler.fit_transform(training_data_df)
# transform scales the data same amount it did with fit_transform
scaled_testing = scaler.transform(testing_data_df)

# print out the adjustment that the scaler applied to the total_earnings column of data
# this will help when we want to rescale the predictions back to their original units
print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8],scaler.min_[8]))

# Create new pandas DataFrame objects from the scaled data
scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=testing_data_df.columns.values)

# Save scaled data dataframes to new CSV files
scaled_training_df.to_csv("Ex_Files_Building_Deep_Learning_Apps/03/sales_data_train_scaled.csv",index=False)
scaled_testing_df.to_csv("Ex_Files_Building_Deep_Learning_Apps/03/sales_data_test_scaled.csv",index=False)

X_train = scaled_training_df.drop('total_earnings', axis=1).values
y_train = scaled_training_df[['total_earnings']].values

# Define the model
# try and test the number of layers
model = Sequential()
model.add(Dense(70, activation='relu', input_shape=(9,)))
model.add(Dense(125, activation='relu'))
model.add(Dense(70, activation='relu'))
# since we are predicting a price we use a linear function
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.summary()

# Train the model
# too few epochs might not train the model well. too many might cause over fitting
# nn train best when data is shuffled
# verbose prints more detail during training so we can watch what's going on
# shuffle helps NN train better
model.fit(X_train, y_train, epochs=70, shuffle=True, verbose=2)


X_test = scaled_testing_df.drop('total_earnings', axis=1).values
y_test = scaled_testing_df[['total_earnings']].values

# to measure the error rate of the testing data
test_error_rate = model.evaluate(X_test, y_test, verbose=0)
# the result will be the error rate for the test data as measured by our cost function
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

# this model is now trained to look at sales and predict sale price
# LOAD THE SEPERATE TEST DATA SET
prop_test_data_df = pd.read_csv("Ex_Files_Building_Deep_Learning_Apps/04/proposed_new_product.csv")

# Make a prediction with the NN
prediction = model.predict(prop_test_data_df)

# Grab just the first element of the first prediction (this is a thing of keras, it thinks we'll ask for multiple predictions)
prediction = prediction[0][0]

# re-scale the data from 0-1 back to dollars
prediction = prediction + abs(scaler.min_[8])
prediction = prediction / scaler.scale_[8]

print("Earnings Prediction for Proposed Product - ${:.2f}".format(prediction))

# Save the model to disk - saves the structure and the weights that determine how the NN works
model.save("trained_model.h5")
print("model saved to disk")

model = load_model("trained_model.h5")

# Make a prediction with the NN
prediction = model.predict(prop_test_data_df)

# Grab just the first element of the first prediction (this is a thing of keras, it thinks we'll ask for multiple predictions)
prediction = prediction[0][0]

# re-scale the data from 0-1 back to dollars
prediction = prediction + abs(scaler.min_[8])
prediction = prediction / scaler.scale_[8]

print("Earnings Prediction for Proposed Product - ${:.2f}".format(prediction))
