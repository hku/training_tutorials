from __future__ import absolute_import, division, print_function

import os.path
import collections
import gzip
import numpy

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

def get_next_dict(data_set, batch_size, images_pl, labels_pl):
	images_feed, labels_feed = data_set.next_batch(batch_size)
	return {
		images_pl: images_feed,
		labels_pl: labels_feed,
	}

def read_data_sets(data_dir, validation_size=5000):

	TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
	TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
	TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
	TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
	
	filepath = os.path.join(data_dir, TRAIN_IMAGES)
	with open(filepath, 'rb') as f:
		train_images = extract_images(f)
	
	filepath = os.path.join(data_dir, TRAIN_LABELS)
	with open(filepath, 'rb') as f:
		train_labels = extract_labels(f)

	filepath = os.path.join(data_dir, TEST_IMAGES)
	with open(filepath, 'rb') as f:
		test_images = extract_images(f)

	filepath = os.path.join(data_dir, TEST_LABELS)
	with open(filepath, 'rb') as f:
		test_labels = extract_labels(f)

	validation_images = train_images[:validation_size]
	validation_labels = train_labels[:validation_size]
	train_images = train_images[validation_size:]
	train_labels = train_labels[validation_size:]
	
	train = DataSet(train_images, train_labels)
	validation = DataSet(validation_images, validation_labels)
	test = DataSet(test_images, test_labels)
	
	print("# of samples, training: %d, validation: %d, test: %d" % (train.num_examples, validation.num_examples, test.num_examples))

	return Datasets(train=train, validation=validation, test=test)

def extract_images(f):
	print('Extracting', f.name)
	with gzip.GzipFile(fileobj=f) as bytestream:
		magic = _read32(bytestream)
		if magic != 2051:
			raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
		num_images = _read32(bytestream)
		rows = _read32(bytestream)
		cols = _read32(bytestream)
		buf = bytestream.read(rows * cols * num_images)
		data = numpy.frombuffer(buf, dtype=numpy.uint8)
		data = data.reshape(num_images, rows, cols, 1)
		return data

def extract_labels(f):
	print('Extracting', f.name)
	with gzip.GzipFile(fileobj=f) as bytestream:
		magic = _read32(bytestream)
		if magic != 2049:
			raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, f.name))
		num_items = _read32(bytestream)
		buf = bytestream.read(num_items)
		labels = numpy.frombuffer(buf, dtype=numpy.uint8)
		return labels

def _read32(bytestream):
	dt = numpy.dtype(numpy.uint32).newbyteorder('>')
	return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

class DataSet(object):
	def __init__(self, images, labels):
		assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
		assert images.shape[3] == 1
		images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])

		images = images.astype(numpy.float32)
		images = numpy.multiply(images, 1.0 / 255.0)
		
		self._images = images
		self._labels = labels
		self._num_examples = images.shape[0]
		self._epochs_completed = 0
		self._index_in_epoch = 0

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size, shuffle=True):
		start = self._index_in_epoch
		if self._epochs_completed == 0 and start == 0 and shuffle:	
			perm0 = numpy.arange(self._num_examples)
			numpy.random.shuffle(perm0)
			self._images = self.images[perm0]
			self._labels = self.labels[perm0]

		if start + batch_size <= self._num_examples:
			self._index_in_epoch += batch_size
			end = self._index_in_epoch
			return self._images[start:end], self._labels[start:end]
		else:
			self._epochs_completed += 1
			rest_num_examples = self._num_examples - start
			images_rest_part = self._images[start:self._num_examples]
			labels_rest_part = self._labels[start:self._num_examples]
			# Shuffle the data
			if shuffle:
				perm = numpy.arange(self._num_examples)
				numpy.random.shuffle(perm)
				self._images = self.images[perm]
				self._labels = self.labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size - rest_num_examples
			end = self._index_in_epoch
			images_new_part = self._images[start:end]
			labels_new_part = self._labels[start:end]
			return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)