# command tensorboard --logdir=06\logs
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense

RUN_NAME = "run 2 with 100 nodes"

training_data_df = pd.read_csv("D:\\Ex_Files_Building_Deep_Learning_Apps\\06\\sales_data_training_scaled.csv")

X = training_data_df.drop('total_earnings', axis=1).values
Y = training_data_df[['total_earnings']].values

# Define the model
model = Sequential()
model.add(Dense(50, input_dim=9, activation='relu', name='layer_1'))
model.add(Dense(100, activation='relu', name='layer_2'))
model.add(Dense(50, activation='relu', name='layer_3'))
model.add(Dense(1, activation='linear', name='output_layer'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Create a TensorBoard logger
logger = keras.callbacks.TensorBoard(
    log_dir="D:\\Ex_Files_Building_Deep_Learning_Apps\\06\\logs\\{}".format(RUN_NAME),
    write_graph=True,
    histogram_freq=0

)

# Train the model
# Callbacks is a list of functions we want keras to call each time it 
# makes a pass through the training data during the training process.
model.fit(
    X,
    Y,
    epochs=100,
    shuffle=True,
    verbose=2,
    callbacks=[logger]
)

# Load the separate test data set
test_data_df = pd.read_csv("D:\\Ex_Files_Building_Deep_Learning_Apps\\06\\sales_data_test_scaled.csv")

X_test = test_data_df.drop('total_earnings', axis=1).values
Y_test = test_data_df[['total_earnings']].values

test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))