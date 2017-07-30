#coding=utf-8  

import numpy as np
import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data

# 加载数据 one_hot encoding 独热编码
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

# 设置训练集
X_train, Y_train = mnist.train.next_batch(1000)

# 设置测试集
X_test, Y_test = mnist.test.next_batch(200)

# 设置变量
x_train = tf.placeholder(tf.float32, [None, 784])
x_test = tf.placeholder(tf.float32, [784])

# 计算距离 reduction_indices=1 表示横向纬度 
# 返回值 为所有训练集到测试集的距离
distance = tf.reduce_sum(tf.abs(tf.add(x_train, tf.negative(x_test))), reduction_indices=1)

# 最短距离  
# 返回最短距离 所在的索引
prediction = tf.arg_min(distance, 0)

# 精确度
accuracy = 0

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # 取测试集
    for i in range(len(X_test)):

        # 返回预测的最短 distance 的训练集索引
        nn_index = sess.run(prediction, feed_dict={x_train: X_train, x_test: X_test[i, :]})
        
        #根据训练集预测索引 取出训练数据的值 
        print("Test", i, "Prediction", np.argmax(Y_train[nn_index]), "True class:", np.argmax(Y_test[i]))

        if np.argmax(Y_train[nn_index]) == np.argmax(Y_test[i]):
            accuracy += 1./len(X_test)
    
    print "Done!"
    print "Accuracy", accuracy
