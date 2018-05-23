import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print("TensorFlow Version: {}".format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
	warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
	# TODO: Implement function
	#   Use tf.saved_model.loader.load to load the model and weights
	vgg_tag = 'vgg16'
	vgg_input_tensor_name = 'image_input:0'
	vgg_keep_prob_tensor_name = 'keep_prob:0'
	vgg_layer3_out_tensor_name = 'layer3_out:0'
	vgg_layer4_out_tensor_name = 'layer4_out:0'
	vgg_layer7_out_tensor_name = 'layer7_out:0'

	#(Q&A)
	tf.save_model_loader.load(sess, [vgg_tag], vgg_tag)
	graph = tf.get_default_graph()
	w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
	keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
	w3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
	w4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
	w7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

	return w1, keep, w3, w4, w7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
	"""
	Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
	:param vgg_layer3_out: TF Tensor for VGG Layer 3 output
	:param vgg_layer4_out: TF Tensor for VGG Layer 4 output
	:param vgg_layer7_out: TF Tensor for VGG Layer 7 output
	:param num_classes: Number of classes to classify
	:return: The Tensor for the last layer of output
	"""
	# Layer 1 in Decoder
	# decoding from layer 7 (1x1 conv)
	decoder1_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

	# Layer 2 in Decoder
	# 1. decoding from layer 4 (1x1 conv)
	decoder2_in1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
	# 2. Upsample from decoder 1
	decoder2_in2 = tf.layers.conv2d_transpose(decoder1_out, num_classes, 4, 2, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
	# Result = 1 + 2
	decoder2_out = tf.add(decoder2_in1, decoder2_in2)

	# Layer 3 in Decoder
	# decoding from layer 3 (1x1 conv)
	decoder3_in1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
	# Upsample from decoder 2
	decoder3_in2 = tf.layers.conv2d_transpose(decoder2_out, num_classes, 4, 2, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
	# Result = 1 + 2
	decoder3_out = tf.add(decoder3_in1, decoder3_in2)

	output = tf.layers.conv2d_transpose(decoder3_out, num_classes, 16, 8, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

	return decoder3_out
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
	"""
	Build the TensorFLow loss and optimizer operations.
	:param nn_last_layer: TF Tensor of the last layer in the neural network
	:param correct_label: TF Placeholder for the correct label image
	:param learning_rate: TF Placeholder for the learning rate
	:param num_classes: Number of classes to classify
	:return: Tuple of (logits, train_op, cross_entropy_loss)
	"""
	# TODO: Implement function
	logits = tf.reshape(nn_last_layer, (-1, num_classes))
	ground_truth = tf.reshape(correct_label, (-1, num_classes))

	# calculate cross entropy loss
	cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= ground_truth))

	# Implement Adam Optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
	train_op = optimizer.minimize(cross_entropy_loss)

	return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
	"""
	Train neural network and print out the loss during training.
	:param sess: TF Session
	:param epochs: Number of epochs
	:param batch_size: Batch size
	:param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
	:param train_op: TF Operation to train the neural network
	:param cross_entropy_loss: TF Tensor for the amount of loss
	:param input_image: TF Placeholder for input images
	:param correct_label: TF Placeholder for label images
	:param keep_prob: TF Placeholder for dropout keep probability
	:param learning_rate: TF Placeholder for learning rate
	"""
	# TODO: Implement function
	sess.run(tf.global_variables_initializer())

	for epoch in epochs:
		i=1
		for image,label in get_batches_fn(batch_size):
			_, loss = sess.run([train_op,cross_entropy_loss], feed_dict={input_image:image , correct_label:correct_label, keep_prob:0.4, learning_rate:learning_rate})
			print("Epoch :{}, Batch :{}, loss:{:.3f}".format(epoch+1,i,loss))
			i=i+1
		print("----------------------------------------------------------------------------")

tests.test_train_nn(train_nn)


def run():
	num_classes = 2
	image_shape = (160, 576)
	data_dir = './data'
	runs_dir = './runs'
	tests.test_for_kitti_dataset(data_dir)

	# Download pretrained vgg model
	helper.maybe_download_pretrained_vgg(data_dir)

	# OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
	# You'll need a GPU with at least 10 teraFLOPS to train on.
	#  https://www.cityscapes-dataset.com/

	with tf.Session() as sess:
		# Path to vgg model
		vgg_path = os.path.join(data_dir, 'vgg')
		
		# Create function to get batches
		get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
		
		# TODO: Build NN using load_vgg, layers, and optimize function
		input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
			
		# Initializing placeholder
		correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
		learning_rate = tf.placeholder(tf.float32, name='learning_rate')
		
		epochs = 1
		batch_size = 10
		
		# TODO: Train NN using the train_nn function
		train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,correct_label, keep_prob, learning_rate)

		# TODO: Save inference data using helper.save_inference_samples
		helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

if __name__ == '__main__':
    run()
