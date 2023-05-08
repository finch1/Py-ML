import tensorflow as tf 
# built in utility for retrieving the dataset on the fly
# import loads the utility we will later use both to automatically 
# download the data for us, and to manage and partition it as needed
from tensorflow.examples.tutorials.mnist import input_data

# defining constraints
DATA_DIR = '/tmp/data'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100

# read_data_sets() of the MNIST reading utility dowloads the dataset
# and saves it locally, setting the stage for further use later
# one_hot tells the utility how we want the data to be labeled
data = input_data.read_data_sets(DATA_DIR, one_hot = True)

# a variable is an element manipulated by the computation, while a placeholder, 
# has to be supplied when triggering it. 
# the image itself (x) is a placeholder, supplied when running the computation graph
# the size[none, 768] means :
# none: is an indicator that we are not currently specifying how many of these images will be used at once
# 28x28 unrolled into a single vector
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10])) # (initial value)

# in supervised learning model, we attempt to learn a model such that the true labels
# and the predicted labels are close in some sense
# elements representing the true and predicted labels 
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

# the measure of similarity we choose for this model is what is known as cross entropy
# - a natural choice when the model outputs class probabilities
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_pred, labels = y_true))

# GDO is the training of the model ( how to minimize the loss function )
# 0.5 is the learning rate, controlling how fast our gradient descent optimizer
# shifts model weights to reduce overall loss
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# evaluation procedure to test the accuracy of the model
# we are interested in the fraction of test exaples that are correctly classified
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

# define a session to make use of the computation graph
# 'with' stmt wraps the execution of a block with methods defined by a context manager-
# -an object that has the special method functions:
# .__enter__() to set up a block of code and .__exit__() to exit the block. 
with tf.Session() as sess:
    # Train. Initialize all the variables
    sess.run(tf.global_variables_initializer())
    # the actual training of the model, in the gradient decent approach, consists of
    # taking many steps in "the right direction." NUM_STEPS is the number of steps made
    # each step, we ask our data manager for a bunch of examples with their labels
    # and present them to the learner. MINIBATCH_SIZE constant controls the number
    # of examples to use for each step. 
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        # when constructing the model, a placeholder was defined. each time we
        # want to run a computation that will include these elements, we must supply a 
        # value for them
        sess.run(gd_step, feed_dict = {x: batch_xs, y_true: batch_ys})

    # Test
    # to evaluate the model we have just finished teaching, we run the accuracy computing
    # operation defined earlier ( accuracy is defined as the fraction of images that are correctly labled)
    # here, we find a separate group of test images, which were never seen by the model during training
    feed_dict={x: data.test.images, y_true: data.test.labels}
    ans = sess.run(accuracy, feed_dict)

print( "Accuracy: {:.4}%".format(ans*100))
