from __future__ import absolute_import, division, print_function

import math
import tensorflow as tf


IMAGE_SIZE = 28

IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

HIDDEN1_UNITS = 128
HIDDEN2_UNITS = 32

NUM_CLASSES = 10

def inference(images):
	# Hidden 1
	with tf.name_scope('hidden1'):
		weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, HIDDEN1_UNITS], stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))), name='weights')
		biases = tf.Variable(tf.zeros([HIDDEN1_UNITS]), name='biases')
		hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

	# Hidden 2
	with tf.name_scope('hidden2'):
		weights = tf.Variable(tf.truncated_normal([HIDDEN1_UNITS, HIDDEN2_UNITS], stddev=1.0 / math.sqrt(float(HIDDEN1_UNITS))), name='weights')
		biases = tf.Variable(tf.zeros([HIDDEN2_UNITS]), name='biases')
		hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

	# Linear
	with tf.name_scope('softmax_linear'):
		weights = tf.Variable(tf.truncated_normal([HIDDEN2_UNITS, NUM_CLASSES], stddev=1.0 / math.sqrt(float(HIDDEN2_UNITS))), name='weights')
		biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
		logits = tf.matmul(hidden2, weights) + biases
	tf.summary.histogram('logits', logits)	
	return logits

def loss(logits, labels):
	labels = tf.to_int64(labels)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
	return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss, learning_rate):
	tf.summary.scalar('loss', loss)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	global_step = tf.Variable(0, name='global_step', trainable=False)
	train_op = optimizer.minimize(loss, global_step=global_step)
	return train_op

def evaluation(logits, labels):
	correct = tf.nn.in_top_k(logits, labels, 1)
	return tf.reduce_sum(tf.cast(correct, tf.int32))
