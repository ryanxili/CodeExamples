import numpy as np
import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

X_train, Y_train = mnist.train.next_batch(1000)

X_test, Y_test = mnist.test.next_batch(200)

x_train = tf.placeholder(tf.float32, [None, 784])
x_test = tf.placeholder(tf.float32, [784])

distance = tf.reduce_sum(tf.abs(tf.add(x_train, tf.negative(x_test))), reduction_indices=1)

prediction = tf.arg_min(distance, 0)

accuracy = 0

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(len(X_test)):

        nn_index = sess.run(prediction, feed_dict={x_train: X_train, x_test: X_test[i]})
        print("index,", nn_index)
        print("Test", i, "Prediction", np.argmax(Y_train[nn_index]), "True class:", np.argmax(Y_test[i]))

        if np.argmax(Y_train[nn_index]) == np.argmax(Y_test[i]):
            accuracy += 1./len(X_test)
    
    print "Done!"
    print "Accuracy", accuracy

    print("y_train,", Y_train)
