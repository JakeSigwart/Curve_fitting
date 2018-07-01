#The purpose of this program is to show that a neural network is a universal approximator
#In order to be a univaersal approximator, the network must have non-linearities i.e. softplus function
#I am training the network to approximate sin x  on [-2*pi, 2*pi]
#Results: 
import os
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

path = os.path.dirname(__file__)
save_path = path + "\\univ_softplus\\classifier.ckpt"

sess = tf.Session()
inputs = tf.placeholder(tf.float32, shape=[None, 1], name='inputs')
labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels')

inputs_1 = tf.map_fn(lambda x_val: tf.fill([64], 1.000)*x_val, inputs)

W_1 = tf.Variable(tf.truncated_normal([64, 64],mean=0.0,stddev=0.088), name='W_1')
b_1 = tf.Variable(tf.constant(0.0005, shape=[64]), name='b_1')
h_1 = tf.matmul(inputs_1, W_1) + b_1

W_2 = tf.Variable(tf.truncated_normal([64, 64],mean=0.0,stddev=0.088), name='W_2')
b_2 = tf.Variable(tf.constant(0.0005, shape=[64]), name='b_2')
h_2 = tf.matmul(h_1, W_2) + b_2
h_actv_2 = tf.nn.softplus(h_2, name='h_actv_2')

W_3 = tf.Variable(tf.truncated_normal([64, 64],mean=0.0,stddev=0.088), name='W_3')
b_3 = tf.Variable(tf.constant(0.0005, shape=[64]), name='b_3')
h_3 = tf.matmul(h_actv_2, W_3) + b_3
h_actv_3 = tf.nn.softplus(h_3, name='h_actv_3')

W_4 = tf.Variable(tf.truncated_normal([64,1],mean=0.0,stddev=0.088), name='W_4')
h_output = tf.matmul(h_actv_3, W_4)

loss = tf.reduce_mean(tf.square(h_output - labels))
Optimize = tf.train.AdamOptimizer(5e-3).minimize(loss)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
if os.path.isfile(path + "\\univ_softplus\\checkpoint"):
	saver.restore(sess, save_path)
	print("Model restored.")
else:
	print('Building new model...')

#TRAINING##
num_batches = 1024
batch_size = 1024

for i in range(num_batches):
	#get random sine values
	x = 4*math.pi*np.random.sample(batch_size) - 2*math.pi
	x = x.reshape([batch_size,1])
	y = np.zeros(shape=[batch_size, 1], dtype=float)
	#y = sin of x[]
	for index in range(batch_size):
		y[index,0] = math.sin(x[index])

	gradients, loss_out, output = sess.run([Optimize, loss, h_output], feed_dict={inputs: x, labels: y})
	print(str(x[0]) + ',  ' + str(output[0]))
	print("Batch: " + str(i) + "  Cross Entropy: " + str(loss_out))
	
save_path = saver.save(sess, path + "\\univ_softplus\\classifier.ckpt")
plt.scatter(x, output)
plt.xlabel('Independant Var x')
plt.ylabel('NN Aproximation of sin(x)')
plt.show()
