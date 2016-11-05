import tensorflow as tf
import numpy as np
import cv2

class TfFunc(object):
    def __init__(self):
        """
        self.sess = tf.InteractiveSession()

        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

        self.W = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))
        
        self.sess.run(tf.initialize_all_variables())

        self.y = tf.matmul(self.x, self.W) + self.b

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_))

        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)
        """
        self.sess = tf.InteractiveSession()

        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        self.W = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))

        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])


        # convolution layer
        self.W_conv1 = self.weight_var([5, 5, 1, 32])
        self.b_conv1 = self.bias_var([32])

        # hidden layer
        self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = self.max_pool_2x2(self.h_conv1)

        # convolution layer
        self.W_conv2 = self.weight_var([5, 5, 32, 64])
        self.b_conv2 = self.bias_var([64])

        # hidden layer
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = self.max_pool_2x2(self.h_conv2)

        # fully-connected layer
        self.W_fc1 = self.weight_var([7*7*64, 1024])
        self.b_fc1 = self.bias_var([1024])

        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        # dropout
        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
        
        # Readout Layer
        self.W_fc2 = self.weight_var([1024, 10])
        self.b_fc2 = self.bias_var([10])

        # output
        self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)



        # train
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_conv, self.y_))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        # initialize all variable
        self.sess.run(tf.initialize_all_variables())
    


    def __del__(self):
        self.sess.close()

    
    def weight_var(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_var(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    

    def tf_sess(self, prevbatch, batch):
        """
        self.train_step.run({self.x: prevbatch[0], self.y_: prevbatch[1]})
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print (accuracy.`eval({self.x: batch[0], self.y_: batch[1]}))    
        print (self.sess.run(self.y, feed_dict={self.x: batch[0]}))
        """
        """
        print self.h_conv1
        print self.h_pool1
        print self.h_conv2
        print self.h_pool2
        print self.h_pool2_flat
        """

        
        for i in range(200):
            self.train_step.run(feed_dict={self.x: prevbatch[0], self.y_: prevbatch[1], self.keep_prob: 0.5})
        
        train_accuracy = self.accuracy.eval(feed_dict={self.x: prevbatch[0], self.y_: prevbatch[1], self.keep_prob: 1.0})
        print( train_accuracy )

        print self.sess.run(self.y_conv, feed_dict={self.x: batch[0], self.keep_prob: 1.0})






face_cascade = cv2.CascadeClassifier('/var/www/flask/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('mov_bbb.mp4')
test = TfFunc()
ret, prev_frame = cap.read()
prev_gray = tuple()
savbatch = tuple()

if ret:
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    sample = cv2.resize(prev_gray, (28, 28))
    
    batchar = sample.reshape((1, 784))
    batchar = batchar.astype(np.float32)
    batchar = np.multiply(batchar, 1.0 / 255.0)

    batchlb = np.zeros((1,10))
    batchlb[0,2] = 1 

    savbatch = (batchar, batchlb)

while(1):
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            roi_gray = gray[y:y+h, x:x+h]

            sample = cv2.resize(roi_gray, (28,28))
            
            batchar = sample.reshape((1, 784))
            batchar = batchar.astype(np.float32)
            batchar = np.multiply(batchar, 1.0 / 255.0)

            batchlb = np.zeros((1,10))
            batchlb[0,1] = 1
            
            batch = (batchar, batchlb)

            
            test.tf_sess(savbatch, batch)


            savbatch = batch
            


        prev_gray = gray.copy()
    else:
        break


    
    
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

