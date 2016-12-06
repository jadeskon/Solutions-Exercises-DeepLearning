# Convolutional Neural Network (CNN) example in TensorFlow
#
# Here we construct a simple Convolutional Neural Network (CNN)
# using TensorFlow (TF) which will learn using the MNIST training dataset
# to classify 28x28 pixel images of digits 0,...,9
#
# Network structure is:
# INPUT --> CONV1/MAXPOOL --> CONV2/MAXPOOL --> FC --> OUT
#
# ---
# by Prof. Dr. Juergen Brauer, www.juergenbrauer.org
 
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from random import randint
 
# 1. get the MNIST training + test data
# Note: this uses the mnist class provided by TF for a convenient
#       access to the data in just a few lines of code
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
 
# show an example of a train image
img_nr = 489
label_vec = mnist.train.labels[img_nr]
print("According to the training data the following image is a ", np.argmax(label_vec) )
tmp = mnist.train.images[img_nr]
tmp = tmp.reshape((28,28))
plt.imshow(tmp, cmap = cm.Greys)
plt.show()
 
 
# 2. set up training parameters
learning_rate = 0.01
training_iters = 20000
batch_size = 128
display_step = 10
 
 
# 3. set up CNN network parameters
n_input      = 784  # MNIST data input dimension (img has shape: 28*28 pixels)
n_classes    = 10   # MNIST nr of total classes (0-9 digits)
dropout_rate = 0.75 # probability to keep an unit in FC layer during training
 
 
# 4. define TF graph input nodes x,y,keep_prob
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

 
# 5. define a helper function to create a single CNN layer
#    with a bias added and RELU function output
def conv2d(x, W, b, strides=1):

    # Conv2D wrapper, with bias and relu activation
    #
    # from: https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#conv2d
    #
    # tf.nn.conv2d(input, filter, strides, padding,
    #              use_cudnn_on_gpu=None, data_format=None, name=None)
    #
    # Computes a 2-D convolution given 4-D input and filter tensors.
    # Given an input tensor of shape
    #    [batch, in_height, in_width, in_channels]
    # and a filter / kernel tensor of shape
    #    [filter_height, filter_width, in_channels, out_channels],
    # this op performs the following:
    #
    #  1. Flattens the filter to a 2-D matrix with shape
    #     [filter_height * filter_width * in_channels, output_channels].
    #  2. Extracts image patches from the input tensor to form a virtual
    #     tensor of shape
    #     [batch, out_height, out_width, filter_height * filter_width * in_channels].
    #  3. For each patch, right-multiplies the filter matrix and the
    #     image patch vector.

    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
 
 
# 6. define a helper function to create a single maxpool operation
#    for the specified tensor x - with a max pooling region of 2x2 'pixels'
def maxpool2d(x, k=2):

    # MaxPool2D wrapper
    # Non overlapping pooling
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
 
 
# 7. helper function to create a CNN model
def conv_net(x, weights, biases, dropout):
 
    # reshape input picture which has size 28x28 to a 4D vector
    # -1 means: infer the size of the corresponding dimension
    # here: it will result in 1
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
 
    # create first convolution layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
 
    # then add a max pooling layer for down-sampling on top of conv1
    conv1 = maxpool2d(conv1, k=2)
 
    # create second convolution layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
 
    # then add a max pooling layer for down-sampling on top of conv2
    conv2 = maxpool2d(conv2, k=2)
 
    # create a fully connected layer
    # thereby: reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
 
    # apply dropout during training for this fully connected layer fc1
    fc1 = tf.nn.dropout(fc1, dropout_rate)
 
    # add output layer: out=fc1*out_weights+out_biases
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
 
    # return tensor operation
    return out
 
 
# 8. initialize layers weights & biases normally distributed and
#    store them in a dictionary each
weights = {
    # 5x5 conv filter, 1 input (depth=1), 32 outputs (depth=32)
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),

    # 5x5 conv filter, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),

    # fully connected, 7*7*64 inputs, 1024 outputs
    # 7x7 is the spatial dimension of CONV2, 64 is its depth
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),

    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
 
biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
 
 
# 9. construct model using helper function
pred = conv_net(x, weights, biases, keep_prob)
 
 
# 10. define error function and optimizer
error_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error_func)
 
 
# 11. evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
 
 
# 12. initializing the variables and init graph
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)
 
 
# 13. keep training until we reached max iterations
step = 1
while step * batch_size < training_iters:
 
        # get next training batch
        batch_x, batch_y = mnist.train.next_batch(batch_size)
 
        # set inputs & run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x,
                                       y: batch_y,
                                       keep_prob: 1.0-dropout_rate}
                )
 
        if step % display_step == 0:
 
            # calculate batch loss and accuracy
            batch_error, acc = \
              sess.run([error_func, accuracy], feed_dict={x: batch_x,
                                                          y: batch_y,
                                                          keep_prob: 1.}
                                       )
 
            print("Iter " + str(step*batch_size) + ", Batch error= " + \
                  "{:.6f}".format(batch_error) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
 
        step += 1
 
print("Optimization finished!")
 
# 14. calculate accuracy for test images
test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images[:990],
                                              y: mnist.test.labels[:990],
                                              keep_prob: 1.0} )


print("Testing Accuracy:", test_accuracy)
 
# 15. show an example of a test image used for computing the accuracy
img_nr = randint(0, 512)
tmp = mnist.test.images[img_nr]
tmp = tmp.reshape((28,28))
plt.imshow(tmp, cmap = cm.Greys)
plt.show()
