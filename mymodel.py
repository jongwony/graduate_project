import tensorflow as tf
import numpy as np
import math

class MySimpleModel(object):
    def __init__(self, resize, label_size):
        # session init
        self.sess = tf.InteractiveSession()

        # variable
        self.x = tf.placeholder(tf.float32, shape=[None, resize*resize])
        self.y_ = tf.placeholder(tf.float32, shape=[None, label_size])

        self.W = tf.Variable(tf.zeros([resize*resize, label_size]))
        self.b = tf.Variable(tf.zeros([label_size]))

        # output
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)

        # train value
        self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y)) 
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)
        
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

       

        # variable initialize
        self.sess.run(tf.initialize_all_variables())

    def __del__(self):
        self.sess.close()

    
    def simple_train(self, pre_bat, bat):
        for i in range(4):
            self.train_step.run({self.x: pre_bat[0], self.y_: pre_bat[1]})
        

        print self.accuracy.eval(feed_dict={self.x: bat[0], self.y_: bat[1]})
        
        
        
        res = self.y.eval(feed_dict={self.x: bat[0]})
        maxidx = self.sess.run(tf.argmax(res, 1)[0])
        
        print res, maxidx
        return maxidx, res[0, maxidx]
   

    def backpropa_train(self, bat):
        for i in range(4):
            self.train_step.run({self.x: bat[0], self.y_: bat[1]})
   

    def feedforward(self, tfimage):
        res = self.y.eval(feed_dict={self.x: tfimage})
        maxidx = self.sess.run(tf.argmax(res, 1)[0])
        print '%f %' % res[0, maxidx]
        return maxidx, res[0, maxidx]
        


class MyTfModel(object):

    def __init__(self, resize, label_size, conv):
        # session
        self.sess = tf.InteractiveSession()

        # variable
        self.x = tf.placeholder(tf.float32, shape=[None, resize*resize])
        self.y_ = tf.placeholder(tf.float32, shape=[None, label_size])
        self.W = tf.Variable(tf.zeros([resize*resize, label_size]))
        self.b = tf.Variable(tf.zeros([label_size]))

        self.x_img = tf.reshape(self.x, [-1, resize, resize, 1])

        # convolution layer, 32 output
        self.W_conv = self.weight_var([conv, conv, 1, 32], resize)
        self.b_conv = self.bias_var([32])

        # hidden layer
        self.h_conv = tf.nn.relu(self.conv2d(self.x_img, self.W_conv) + self.b_conv)
        self.h_pool = self.max_pool_2x2(self.h_conv)

        # fully-connected layer, 1024 neuron
        self.W_fc = self.weight_var([resize*resize/4 * 32, 1024], resize)
        self.b_fc = self.bias_var([1024])
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, resize*resize/4 * 32])
        self.h_fc = tf.nn.relu(tf.matmul(self.h_pool_flat, self.W_fc) + self.b_fc)

        # readout layer
        self.W_ro = self.weight_var([1024, label_size], resize)
        self.b_ro = self.bias_var([label_size])

        # output
        self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc, self.W_ro) + self.b_ro)

        # training
        self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y_conv))
        self.train_step= tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)

        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # graph initialize
        self.sess.run(tf.initialize_all_variables())

        
        


    def __del__(self):
        self.sess.close()




    def weight_var(self, shape, resize):
        initial = tf.truncated_normal(shape, stddev=(1.0 / float(resize*resize)))
        return tf.Variable(initial)




    def bias_var(self, shape):
        initial = tf.constant(0.001, shape=shape)
        return tf.Variable(initial)




    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')




    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')




    def tf_train(self, pre_bat, bat):
        for i in range(4):
            self.train_step.run(feed_dict={self.x: pre_bat[0], self.y_:pre_bat[1]})

        print self.accuracy.eval(feed_dict={self.x: bat[0], self.y_: bat[1]})
        
        res = self.y_conv.eval(feed_dict={self.x: bat[0]})
        maxidx = self.sess.run(tf.argmax(res, 1)[0])
        
        print res, maxidx 

        return maxidx, res[0, maxidx]


    def backpropa_train(self, bat):
        for i in range(3):
            self.train_step.run({self.x: bat[0], self.y_: bat[1]})
   
    

    def feedforward(self, tfimage):
        res = self.y.eval(feed_dict={self.x: tfimage})
        maxidx = self.sess.run(tf.argmax(res, 1)[0])
        print '%f %' % res[0, maxidx]
        return maxidx, res[0, maxidx]
