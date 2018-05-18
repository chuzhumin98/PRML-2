from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

# 定义网络超参数
learning_rate = 0.001
training_iters = 200000
#batch_size = 128
batch_size = tf.placeholder(tf.int32, [])
display_step = 10

# 定义网络参数
n_inputs = 28  # 输入的维度
n_steps = 28
n_hidden_units = 128  # 隐藏层的神经元个数
n_classes = 10  # 输出的数量，也就分类数量，0-9

# 占位符输入
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

weights = {'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
           'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))}

biases = {'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
          'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))}

# 构建RNN模型
X = tf.reshape(x, [-1, n_inputs])
X_in = tf.matmul(X, weights['in']) + biases['in']
X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
# 使用基本的LSTM循环网络单元
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
# 初始化为0，LSTM单元由两部分构成(c_state, h_state)
init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
# dynamic_rnn接收张量要么为(batch, steps, inputs)或者(steps, batch, inputs)作为X_in
outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

pred = tf.matmul(final_state[1], weights['out']) + biases['out']

# 定义损失函数和学习步骤
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 测试网络
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化所有的共享变量
init = tf.global_variables_initializer()

# 开启一个训练
with tf.Session() as sess:
    sess.run(init)
    step = 0
    batch_size0 = sess.run(batch_size, feed_dict={batch_size:128})
    # Keep training until reach max iterations
    while step * batch_size0 < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size0)
        batch_xs = batch_xs.reshape([batch_size0, n_steps, n_inputs])
        # 获取批数据
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, batch_size:batch_size0})
        if step % display_step == 0:
            # 计算精度
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1., batch_size:batch_size0})
            # 计算损失值
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1., batch_size:batch_size0})
            print("Iter " + str(step * batch_size0) + ", Minibatch Loss= " + "{:.6f}".format(loss) +
                  ", Training Accuracy = " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    # 计算测试精度
    test_data = mnist.test.images.reshape((-1, n_steps, n_inputs))
    test_labels = mnist.test.labels
    batch_size0 = len(test_labels)
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data,
                                                             y: test_labels,
                                                             keep_prob: 0.5, batch_size:batch_size0}))
    print('**********************')
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data,
                                                             y: test_labels,
                                                             keep_prob: 1.0, batch_size:batch_size0}))