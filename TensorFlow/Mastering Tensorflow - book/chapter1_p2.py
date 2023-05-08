"""
A data flow graph or computation graph is the basic unit of computation in TF 
and is made up of nodes and edges. Each node represents an operation (tf.Operation)
and each edge represents a tensor (tf.Tensor) that gets transferred between the 
nodes.

TF comes with default graph. Unless another graph is explicitly specified, 
a new node (operation) gets implicitly added to the default graph.
"""

import tensorflow as tf 

tfs = tf.InteractiveSession()

graph = tf.get_default_graph()

x1 = tf.constant(5)
x2 = tf.constant(5)
x3 = tf.constant(5)

y = x1 + x2 + x3
sum = tfs.run(y)
print("sum: ", sum)

tfs.close()

# assume linear model y = w * x + b
# define model parameters
w = tf.Variable([.2], dtype=tf.float32)
b = tf.Variable([.4], dtype=tf.float32)

# define model input and output
x = tf.placeholder(tf.float32)
y = (w * x) + b
output = 0

"""
creatung and using a session in the with block ensures that the session is 
automatically closed when the block is finished. otherwise use tfs.close()"""

with tf.Session() as tfs:
    # initialize and print the variable y
    tf.global_variables_initializer().run()
    output = tfs.run(y, feed_dict={x:[1,2,3,4]})
print("\nOutput: ", output)

"""
nodes are executed in order of dependency. i.e. if a depends on b, a executes first
to control the order of execution use tf.Graph.contol_dependencies()
to execute c and d before a and b, 
    with graph_variable.control_dependencies([c,d]):
        # other statements here

this makes sure that any node in the precedeing with block is executed only after c and d are """

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())