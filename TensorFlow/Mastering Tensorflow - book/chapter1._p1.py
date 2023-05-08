"""
TensorFlow can be described with data model, programming model and execution model.

Tensors are n-dim collection of data identified by rank, shape and type
Rank: the number of dimensions of a tensor
Shape: the list denoting the size in each dimension
Scalar - rank 0, shape [1]
Vector - rank 1, shape [col] or [row]
Matrix - rank 2, shape [row, col]
Tensor - rank 3, shape [axis 0, axis 1, axis 2]

Data type: type of data tensor can only store
"""

import tensorflow as tf 

# the only between session() and interactivesession() is that the session created with
# interactivesession becomes the default session. Thus we do not need to specify the 
# session context to execute the session-related command later
tfs = tf.InteractiveSession()

# we have a constant object hello. The tfs object can evaluate hello with hello.eval().
# if tfs was a session() object, then we have to use wither tfs.hello.eval()
# or a with block. The most common practice is to use the with block. 

hello = tf.constant("Hello TensorFlow!!")

print(tfs.run(hello))

# Different ways to create constant values : providing a value at the time of defining the tensor
# tf.constant(value, dtype, shape, name, verify_shape)
c1 = tf.constant(
    value = 5,
    name = 'x'
)

c2 = tf.constant(
    value = 6.,
    name = 'y'
)

c3 = tf.constant(
    value = 7.,
    name = 'z',
    dtype=tf.float32
)
# print the constants
print(c3)
# print values of constants
print(tfs.run([c1, c2, c3]))

# some tf operations
op1 = tf.add(c2, c3)
op2 = tf.multiply(c2, c3)
# observe that the operations are defined as tensors
print("op1: ", op1)
print("op2: ", op2)
print("run op1: ", tfs.run(op1))
print("run op2: ", tfs.run(op2))

# placeholders : create tensors whose values can be provided at runtime. 
# tf.placeholder(dtype, name, shape)
p1 = tf.placeholder(tf.float32)
p2 = tf.placeholder(tf.float32)
print("p1: ", p1)
print("p2: ", p2)

op3 = p1 * p2
# run command in tf session, feeding dictionary (the second arg to the run() op)
# with values for p1 and p2
print("run(op3, feed_dict = {p1:2., p2:3.}): ", tfs.run(op3, feed_dict = {p1:2., p2:3.}))
# the elements of the two input vectors are multiplied in an element wise fashion
print("run(op3, feed_dict = {p1:[2., 3., 4.] , p2:[1., 2., 3.] }): ", tfs.run(op3, feed_dict = {p1:[2., 3., 4.] , p2:[1., 2., 3.]}))

# create tensors from python objects such as a list and NumPy arrays
# tf.convert_to_tensor(value, dtype, name, prefrred_dtype)

# variables: while building and training models, values can be stored in memory
# and updated at runtime. 
# Variables are tensor objects
"""
placeholders: defines input data that does not change over time
    & does not need to be initialized at definition
variables: defines data modified over time
    & needs to be initialized at definition
"""

# linear model ex. y = Wx + b
# w & b must be initialized and shall be adjusted by time
w = tf.Variable([.3], tf.float32, name='w')
b = tf.Variable([-.3], tf.float32, name='b')

# x being feed data at runtime, has to be a placeholder
x = tf.placeholder(tf.float32, name='x')

# y is an opreation
y = tf.add(tf.multiply(w,x),b)

# getting an idea
print()
print("w:", w)
print("x:", x)
print("b:", b)
print("y:", y)

# as we said before, variables must be initialized before used in a session
# for a single variable : tfs.run(w.initializer)
# for only a set of variables : tfs.run(tf.global_variable_initializer())
# for all variables pre run() of a session object : tf.global_variables_initializer().run()
tfs.run(tf.global_variables_initializer())

# run model for these input values
print("\nrun(y,{x:[1,2,3,4]})", tfs.run(y,feed_dict={x:[1,2,3,4]}))

# generating tensors differently, like from functions
a  = tf.zeros(shape=(1,5), name="a", dtype=tf.int32)
print("\na-zeros: ", a)
print(tfs.run(a))
b = tf.ones(shape=(1,6), name="b", dtype=tf.int32)
print("\nb-ones: ", b)
print(tfs.run(b))

# other tensor generating functions
"""
ones / zeros(shape, name, dtype) = creates tensor of the provided shape with all elements zero
ones / zeros_like(tensor, dtype, name, optimizer) = creates tensor with shape of the argument, all elements to zero
fill(dims, value, name) = creates tensor with shape dims, all elements with value. ex fill([100],0)
"""
c = tf.zeros_like(b, name="c")
print("\nc-zeros_like: ", c)
print(tfs.run(c))
d = tf.fill(dims=(1,7), value=7, name="d")
print("\nd-fill: ", d)
print(tfs.run(d))

# populating tensor elements with sequences
"""
lin_space(start,stop,num,name) = generates a 1-d tensor from a sequence of num numbers, range [start, stop], data type as start arg
    ex. a = tf.lin_space(1,100,10)
        returns [1,12,23,34,45,56,67,78,89,100]

range(start,limit,delta=1,dtype,name)        
 =  1-d tensor 
    sequence of numbers range start -> limit
    with delta increments
    same type as start or dtype
    if start is omitted, starts from 0

ex. a = tf.range(start=1, limit=91, delta=10)
returns [1,11,21,31,41,51,61,71,81]
91 (limit) not included

"""
e = tf.lin_space(start=0., stop=100, num=10, name="e")
print("\ne-lin_space: ", e)
print(tfs.run(e))

f = tf.range(start=0., limit=50, delta=10, name="f")
print("\nf-range: ", f)
print(tfs.run(f))
# populating tensor elements with a random distribution
"""
graph level seed is set using tf.set_random_seed
operation level seed is given as the argument seed in all of the random distribution functions

random_normal(shape, mean=0.0, stddev=1.0, dtype, seed, name)
 = tensor of the specified shape
   filled with values from a normal distribution: normal(mean, stddev)

truncated_normal(shape, mean, stddev, dtype, seed, name)
 = tensor of the specified shape
   filled with values from a normal distribution: normal(mean, stddev)
   truncated means thatthe values returned are always at a distance less than two standard deviations from the mean

random_uniform(shape, minval=0, maxval, dtype, seed, name)
 = tensor of the specified shape
   filled with values from a normal distribution: uniform([minval, maxval))

"""

g = tf.random_normal(shape=(1,10), mean=0.0, stddev=1.0, dtype=tf.float32, name="g")
print("\ng-random_normal: ", g)
print(tfs.run(g))
h = tf.truncated_normal(shape=(1,10), mean=0.0, stddev=1.0, dtype=tf.float32, name="h")
print("\nh-truncated_normal: ", h)
print(tfs.run(h))
i = tf.random_uniform(shape=(1,10), minval=0, dtype=tf.float32, name="i")
print("\ni-random_uniform: ", i)
print(tfs.run(i))

"""
TF throws exception when trying to define a virable already defined. 
So use tf.get_variable() instead of tf.Variable()
get_var returns the existing var with the same name if it exists, and creates the 
variable with the specified shape and initializer if it doesnt exist
"""

w2 = tf.get_local_variable(name='w', shape=(2,2), dtype=tf.float32, initializer=tf.orthogonal_initializer)
print("\nw2: ", w2)

# sharing or reusing variables, suggests the use of: tf.variable_scope.reuse_variable() or tf.variable.scope(reuse=True)