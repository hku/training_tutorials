from __future__ import absolute_import, division, print_function

import sys, os.path, time
import argparse
from six.moves import xrange

import tensorflow as tf
import mnist
import input_data


FLAGS = None #inialized at __main__

def run_training():
	batch_size = 100
	max_steps = 2000
	image_pixels = mnist.IMAGE_PIXELS

	data_sets = input_data.read_data_sets(FLAGS.input_data_dir)
	
	with tf.Graph().as_default():
		images_placeholder = tf.placeholder(shape=[batch_size, image_pixels], dtype=tf.float32)
		labels_placeholder = tf.placeholder(shape=[batch_size], dtype=tf.int32)

		logits = mnist.inference(images_placeholder)
		loss = mnist.loss(logits, labels_placeholder)

		train_op = mnist.training(loss, FLAGS.learning_rate)  

		eval_correct = mnist.evaluation(logits, labels_placeholder)
		summary = tf.summary.merge_all()

		init = tf.global_variables_initializer()

		NUM_CORES = 4;
		sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES))

		summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

		sess.run(init)
		
		start_time = time.time()

		for step in xrange(max_steps):
			feed_dict = input_data.get_next_dict(data_sets.train, batch_size, images_placeholder, labels_placeholder)
			sess.run(train_op, feed_dict=feed_dict)

			if step % 100 == 0:
				summary_str, loss_value = sess.run([summary, loss], feed_dict=feed_dict)

				summary_writer.add_summary(summary_str, step)
				summary_writer.flush()
				
				print('Step %d: loss = %.2f' % (step, loss_value))

		duration = time.time() - start_time
		print("duration: %d sec" % duration)

def main(_):
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	run_training()



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--input_data_dir',
		type=str,
		default='./MNIST_data',
		help='Directory to put the input data.'
	)
	parser.add_argument(
		'--log_dir',
		type=str,
		# default=os.path.dirname(os.path.abspath(__file__)) + '/log',
		default='./log',
		help='Directory to put the log data.'
	)
	parser.add_argument(
		'--learning_rate',
		type=float,
		default=0.01,
		help='Initial learning rate.'
	)

	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main)

