from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

#from tensorflow.nn import rnn, rnn_cell
import numpy as np


print('输入数据打shape:')
print(mnist.train.images.shape)
a= np.asarray(range(20))
b = a.reshape(-1,2,2)
c = np.transpose(b,[1,0,2])
d = c.reshape(-1,2)
'''
To classify images using a reccurent neural network, we consider every image row as a sequence of pixels.
Because MNIST image shape is 28*28px, we will then handle 28 sequences of 28 steps for every sample.
'''
# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 100

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)# tf Graph input
x = tf.placeholder("float32", [None, n_steps, n_input])
# Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
y = tf.placeholder("float32", [None, n_classes])

# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=0.0)
_state = lstm_cell.zero_state(batch_size,tf.float32)
a1 = tf.transpose(x, [1, 0, 2]) #28,N,28
a2 = tf.reshape(a1, [-1, n_input]) #N*28,28
a3 = tf.matmul(a2, weights['hidden']) + biases['hidden'] #N*28
a4 = tf.split(a3, n_steps, axis=0) #这个地方接口有所变化

outputs, states = rnn.static_rnn(lstm_cell, a4, initial_state = _state)
a5 = tf.matmul(outputs[-1], weights['out']) + biases['out']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=a5, labels=y))

#AdamOptimizer
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
correct_pred = tf.equal(tf.argmax(a5,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)
step = 1
# Keep training until reach max iterations
while step * batch_size < training_iters:
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    # Reshape data to get 28 seq of 28 elements
    batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
    # Fit training using batch data
    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
    if step % display_step == 0:
            # Calculate batch accuracy
        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,})
            # Calculate batch loss
        loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
        print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) +  ", Training Accuracy= " + "{:.5f}".format(acc))
    step += 1
print("Optimization Finished!")

test_len = batch_size
test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
test_label = mnist.test.labels[:test_len]
# Evaluate model
correct_pred = tf.equal(tf.argmax(a5,1), tf.argmax(y,1))
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
