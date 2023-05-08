# NODES are OPERATIONS, EDGES are TENSOR OBJECTS
# when we construct a node in the graph, like tf.add(), we are actually
# creating an operation instance. These operations do not produce actual values
# until the graph is executed, but rather reference their to-be-computed result as a 
# handle that can be passed on - FLOW - to another node. These handles are referred to
# as Tensor objects.

# Tensor is the name of an object used in the python api as a handle for the result of an operation in the graph
# 1x1 tensor is a scalar, an 1xn tensor is a vector, a nxn tensor is a matrix, a nxnxn tensor is a 3d array
# right after TF is imported, a specific empty default graph is formed. 
# all the nodes we create are automatically associated with the default graph.
import tensorflow as tf 
import numpy as np 
s = 0
t = tf.constant([   [1,2,3],
                    [4,5,6] ])
w = tf.constant(np.array(   [
                            [[1,2,3],
                            [4,5,6]],
                            [[7,8,9],
                            [0,0,0]]    ]))
x = tf.constant(4, dtype=tf.float64)
y = tf.constant([1,2,3])
z = tf.constant(np.arange(4).reshape(2,2))
print(t) # shape=(2, 3) / format(c.get_shape())
print(w) # shape=(2, 2, 3)
print(x) # shape=() - scalar
print(y) # shape=(3,)
print(z) # shape=(2, 2)
print("The get_shape method returns the shape of the tensor as a tuple of integers\n")
print("3d Numpy array input: {}".format(w.get_shape()))
# the first three nodes are each told to output a constant value.
a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

# the next three nodes gets two existing variables as inputs and performs simple
# arithmetic operations
d = tf.multiply(a, b)
e = tf.add(c, b)
f = tf.subtract(d, e)
# we have created our first TensorFlow graph with a-f

# once done decribing the computation graph, we're ready to run the computations
# that it represents. for this to happen, we need to create and run a session

# first we launch the graph in a tf.Session
# a Session object is the part of the TensorFlow API that communicates between Python
# objects and data on our end, and the actual computational system where memory is 
# allocated for the objects we define, intermediate variables are stored, and results 
# are fetched for us. 
sess = tf.Session()
# execution itself is done with the .run() method of the Session object. 
# this method completes one set of computations in the graph as follows:
# it starts at the requested output(s) and then works backword, 
# computing nodes that must be executed according to the set of dependencies.
outs = sess.run(f) # request for node f to be computed
sess.close() # free resources on completion of computation task
print("\nouts = {}".format(outs))

a = tf.constant(5)
b = tf.constant(25)
c = tf.multiply(a, b)
c = tf.cast(c, tf.float32)
d = tf.sin(c)
d = tf.cast(d, tf.int32)
e = tf.divide(d, b)

sess = tf.Session()
outs = sess.run(e)
sess.close()
print("\nouts = {}".format(outs))

# other example
a = tf.constant(2)
b = tf.constant(4)
c = tf.multiply(a, b)   # 8
d = tf.add(a, b)        # 6
e = tf.subtract(c, d)   # 2
f = tf.add(c, d)        # 14
g = tf.divide(f, e)     # 7        

sess = tf.Session()
outs = sess.run(g)
sess.close()
print("\nouts = {}".format(outs))


# other then the default graph created on 'import', we can create additional graphs 
# and control their association with some given operations.
# tf.Graph() creates a new graph, represented as a TF object. 
print(tf.get_default_graph())

g = tf.Graph() # an empty graph g
a = tf.constant(5)
print(g)
# check which graph is currently set as default
# for a given node, view the graph it's associated with - <node>.graph attribute
print(a.graph is g)
print(a.graph is tf.get_default_graph())
print()
# when working with multiple graphs use the 'with' statement together with the as_defualt() command,
# which returns a context manager that makes this graph the default one

g1 = tf.get_default_graph()
g2 = tf.Graph()

print(g1 is tf.get_default_graph())

# g2 only goes default with the with statement
with g2.as_default():
    print(g1 is tf.get_default_graph())

print(g1 is tf.get_default_graph())

# fetches
# the argument passes to run() is called 'fetches', corresponding to the elements of 
# the graph we wish to compute. Multiple node output request:

a = tf.constant(2)
b = tf.constant(4)
c = tf.multiply(a, b)   # 8
d = tf.add(a, b)        # 6
e = tf.subtract(c, d)   # 2
f = tf.add(c, d)        # 14
g = tf.divide(f, e)     # 7     

# 'with' clause automatically closes the Session
with tf.Session() as sess:
    fetches = [a, b, c, d, e, f, g]
    outs = sess.run(fetches)

print("outs = {}".format(outs))
for i in list(outs):
    print(type(i))

# allows replacing the usual tf.Session()
sess = tf.InteractiveSession()
c = tf.linspace(0.0, 4.0, 5)
print("The content of c : \n {}\n".format(c.eval()))
sess.close()

# matrix multiplication:
# before using matmul(), both consts have the same number of
# dimensions and that they are aligned correctly with 
# respect to the inteded multiplication
A = tf.constant(np.arange(1,7).astype(int).reshape(2,3))
print(A.get_shape())

x = tf.constant([1,0,1])
print(x.get_shape())

sess = tf.InteractiveSession()
print("{}".format(A.eval()))
print("{}".format(x.eval()))
sess.close()
# Add a dimension to x, transforming it from a 1D vector
# to a 2D single-column matrix
x = tf.expand_dims(x,1)
print(x.get_shape())

sess = tf.InteractiveSession()
print("{}".format(x.eval()))
sess.close()

b = tf.matmul(A,x)
print(b.get_shape())
sess = tf.InteractiveSession()
print("matmul result:\n{}".format(b.eval()))
sess.close()

# NAMES
with tf.Graph().as_default():
    c1 = tf.constant(4, dtype=tf.float64, name='c')
    with tf.name_scope("prefix_name"):
        c2 = tf.constant(4, dtype=tf.int32, name='c')
        c3 = tf.constant(4, dtype=tf.float64, name='c')

print(c1.name)
print(c2.name)
print(c3.name)

# the optimization process serves to tune the parameters of some given model.
# For this purpose, TensorFlow uses special objects called Variables.
# Unlike other objects that are "refilled" with data each time we run the session,
# Variables can maintain a fixed state in the graph.
# First we call the tf.Variable() function in order to create a Variable and define
# what value it will be initialized with. Then, explicitly perform an initialization 
# operation by running the session with the tf.global_variables_initializer() method,
# which allocates the memory for the Variable and sets its initial values. 

# Variables are computed only when the model runs
init_val = tf.random_normal((1,5),0,1)
var = tf.Variable(init_val, name='var')
print("pre run: \n{}".format(var))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    post_var = sess.run(var)

print("\npost run: \n{}".format(post_var))

# Placeholder: empty Variables that will be filled with data later on
x_data = np.random.randn(5,10)
w_data = np.random.randn(10,1)

with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, shape=(5,10))
    w = tf.placeholder(tf.float32, shape=(10,1))
    b = tf.fill((5,1),-1.)
    xw = tf.matmul(x,w)

    xwb = xw + b
    # 'reduce' is used becasue we are reducing a fice-unit vector to a single scalar
    s = tf.reduce_max(xwb)
    with tf.Session() as sess:
        outs = sess.run(s, feed_dict={x: x_data, w: w_data})

    print("outs = {}".format(outs))

    