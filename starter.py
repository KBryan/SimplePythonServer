import tensorflow as tf
import numpy as np
from IPython.utils.py3compat import xrange

N = 100
data_X = np.random.randn(N, 3)
data_T = np.zeros((N, 3))
means = np.array([[2, 2, 2], [-2, -2, -2], [2, 2, -2]])

for i in xrange(N):
    k = np.random.randint(3)
    data_X[i] += means[k]
    data_T[i, k] = 1


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, W1, W2):
    Z = tf.nn.sigmoid(tf.matmul(X, W1))
    return tf.matmul(Z, W2)


X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])

W1 = init_weights([3, 5])
W2 = init_weights([5, 3])

py_x = model(X, W1, W2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))

train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(1000):
    sess.run(train_op, feed_dict={X: data_X, Y: data_T})
    true = np.argmax(data_T, axis=1)
    pred = sess.run(predict_op, feed_dict={X: data_X, Y: data_T})
    if i % 10 == 0:
        print(np.mean((true == pred)))
