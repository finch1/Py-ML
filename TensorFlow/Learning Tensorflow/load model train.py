# https://www.lynda.com/Google-TensorFlow-tutorials/Visualize-training-runs/601800/647737-4.html
# tenserboard --logdir=
import tensorflow as tf 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

ROOT_DIR = "C:\\Users\\GOKU\\Documents\\PyLrn\\Learning Tensorflow\\"
DATA_DIR = "C:\\Users\\GOKU\\Documents\\PyLrn\\Ex_Files_Building_Deep_Learning_Apps\\03\\"

# load training data set from CSV file
training_data_df = pd.read_csv(DATA_DIR + "sales_data_training.csv", dtype=float)

# pull out columns for X ( data to train with) and Y ( value to predict)
# axis=1 parameter tells it we want to drop a column not a row
# .values to get back the result as an array
X_training =  training_data_df.drop('total_earnings', axis = 1).values
Y_training =  training_data_df[['total_earnings']].values

# load testing data set from CSV file
test_data_df = pd.read_csv(DATA_DIR + "sales_data_test.csv", dtype=float)

X_test =  training_data_df.drop('total_earnings', axis = 1).values
Y_test =  training_data_df[['total_earnings']].values

# all data needs to be scaled to a small range like 0 to 1 for the NN to 
# work well. Create scalers for the inputs and outputs
X_scaler = MinMaxScaler(feature_range=(0,1))
Y_scaler = MinMaxScaler(feature_range=(0,1))

# scale both the training inputs and outputs
X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

# it is very important that the training and test data are scaled with the same scaler
X_scaled_testing = X_scaler.transform(X_test)
Y_scaled_testing = Y_scaler.transform(Y_test)

print(X_scaled_testing.shape)
print(Y_scaled_testing.shape)

print("Note: Y values where scaled by multiplying by {:.10f} and adding {:.4f}".format(Y_scaler.scale_[0], Y_scaler.min_[0]))

# define model parameters
learning_rate = 0.001
training_epochs = 200 # number of iteration loops to train network
display_step = 5

# define how many inputs and outputs are in our neural network
number_of_inputs = 9
number_of_outputs = 1

# define how many neurons we want in each layer of our neural net
nodes_layer1 = 50
nodes_layer2 = 100
nodes_layer3 = 50

# Section one: define the layers of the neural net itself
"""Variable scopes functions are similar to python functions.
Any variable we create within this scope will automatically get a prefix of input to their name
internally in TF. Everything within the same scope will be grouped together within the diagram"""
# input layer
with tf.variable_scope('input'):
    """NN should accept nine floats as input for making predictions each time.
    placeholder objects args(what type of tensor to accept,
                             size or shape of tensor to expect=(
                             None = NN can mix up batches of any size,
                             expect nine values for each record in the batch))    """
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

# layer 1
with tf.variable_scope('layer1'):
    weights = tf.get_variable(name="weights1", shape=[number_of_inputs, nodes_layer1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())# for each connection between each node and node in previous layer
    biases = tf.get_variable(name="biases1", shape=[nodes_layer1], initializer=tf.zeros_initializer()) # variable holds value over time
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases) # activation function that outputs the result of the layer


# layer 2
with tf.variable_scope('layer2'):
    weights = tf.get_variable(name="weights2" , shape=[nodes_layer1, nodes_layer2], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())# for each connection between each node and node in previous layer
    biases = tf.get_variable(name="biases2", shape=[nodes_layer2], initializer=tf.zeros_initializer()) # variable holds value over time
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases) # activation function that outputs the result of the layer

# layer 3
with tf.variable_scope('layer3'):
    weights = tf.get_variable(name="weights3" , shape=[nodes_layer2, nodes_layer3], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())# for each connection between each node and node in previous layer
    biases = tf.get_variable(name="biases3", shape=[nodes_layer3], initializer=tf.zeros_initializer()) # variable holds value over time
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases) # activation function that outputs the result of the layer

# output layer
with tf.variable_scope('output'):
    weights = tf.get_variable(name="weights4" , shape=[nodes_layer3, number_of_outputs], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())# for each connection between each node and node in previous layer
    biases = tf.get_variable(name="biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer()) # variable holds value over time
    prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases) # activation function that outputs the result of the layer

# to be able to train the model, we need a cost function. Also called a loss function 
# tells us how wrong the neural network is when trying to predict the correct output for 
# a single pice of training data. 

with tf.variable_scope('cost'):
    # a node for the expected value that we'll feed in during training
    # Y has one single output, so shape is none , 1
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

# optimizer function that tensorflow can call to train the network
# when TF executes the optimizer, it should run one iteration of the Adam optimizer
# in an attempt to make the cost value smaller
with tf.variable_scope('train'):
    # cost is the variable we want to minimize
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# create a summary operation to log the progress of the network
with tf.variable_scope('log'):
    tf.summary.scalar('current_cost', cost)
    tf.summary.histogram('predicted_value', prediction)
    # when this special node is run, it executes all the summary nodes in pur graph
    # without having to ecplicitly list them all. 
    summary = tf.summary.merge_all()

saver = tf.train.Saver()

# initialize a session so that we can run TensorFlow operations
with tf.Session() as session:
    # run the global variable initializer to init all variables and layers
    ###session.run(tf.global_variables_initializer())### commented when loading model from checkpoint 
    
    # instead load from disk
    saver.restore(session, ROOT_DIR + "logs\\trained_model.ckpt")
    # create log file writers to record training progress.
    # we'll store training and testing log data separately
    training_writer = tf.summary.FileWriter(ROOT_DIR + "logs\\{}\\training".format(RUN_NAME), session.graph)
    testing_writer = tf.summary.FileWriter(ROOT_DIR + "logs\\{}\\testing".format(RUN_NAME), session.graph)

    # run the optimizer over and over to train the network
    # one epoch is one full run through the training data set
    for epoch in range(training_epochs):
        # feed in the training data and do one step of neural network training
        # Optimizer needs additional data to run; training data and expected results for each training pass
        # a placeholder X holds training data and Y holds expected results
        # to pass these, use feed_dict parameter - Python dictionary
        session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
        
        # every 5 training steps, log our progress
        if epoch % 5 == 0:
            training_cost, training_summary = session.run([cost, summary], feed_dict={X: X_scaled_training, Y: Y_scaled_training})
            testing_cost, testing_summary  = session.run([cost, summary], feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})
            
            # write the current training status to the log file. ( y, x ) on the graph
            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)

            # print the current training status to the screen
            print("Training pass: {}".format(epoch), "Training cost: {:7f}".format(training_cost), "Testing cost: {:7f}".format(testing_cost))

    # training is now complete!
    print("Training Done!")

    final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
    final_testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})

    save_path = saver.save(session, ROOT_DIR + "logs\\trained_model.ckpt")
    print("Final Training cost: {:7f}".format(training_cost), " Final Testing cost: {:7f}".format(testing_cost))
