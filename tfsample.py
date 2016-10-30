import tensorflow as tf
import numpy as np
import cv2

class TfFunc(object):
    def __init__(self):
        self.sess = tf.InteractiveSession()

        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

        self.W = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))
        
        self.sess.run(tf.initialize_all_variables())

        self.y = tf.matmul(self.x, self.W) + self.b

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_))

        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)
        
    def __del__(self):
        self.sess.close()

    """
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
    """

    def tf_sess(self, prevbatch, batch):
        self.train_step.run({self.x: prevbatch[0], self.y_: prevbatch[1]})
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print (accuracy.eval({self.x: batch[0], self.y_: batch[1]}))    
        print (self.sess.run(self.y, feed_dict={self.x: batch[0]}))


face_cascade = cv2.CascadeClassifier('/var/www/flask/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('static/uploads/mov_bbb.mp4')
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

