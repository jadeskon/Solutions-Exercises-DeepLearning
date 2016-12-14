# file: inference.py
#
# Builds up the AlexNet model in TensorFlow.
#
# AlexNet was presented in a paper in 2012
# and was the winner of the ILSVRC 2012 competition.
# It was one of the reasons for the Deep Learning Tsunami.
#
# Details can be found in the original publication:
# 
# Krizhevsky, A.; Sutskever, I. & Hinton, G. E.:
#    ImageNet Classification with Deep Convolutional Neural Networks
#    Advances in Neural Information Processing Systems 25
#    Curran Associates, Inc., 2012, 1097-1105
#
# The implementation found here is strongly based on the
# AlexNet implementation by Wang Xinbo:
#    https://github.com/SidHard/tfAlexNet
# 
# ---
# Prof. Dr. Juergen Brauer, www.juergenbrauer.org


import tensorflow as tf


# helper function to build 1st conv layer with filter size 11x11
# and stride 4 (in both directions) and no padding
def conv1st(name, l_input, filter, b):
    cov = tf.nn.conv2d(l_input, filter, strides=[1, 4, 4, 1], padding='VALID')
    return tf.nn.relu(tf.nn.bias_add(cov,b), name=name)
    
# in all other layers we use a stride of 1 (in both directions)
# and a padding such that the spatial dimension (width,height)
# of the output volume is the same as the spatial dimension
# of the input volume
def conv2d(name, l_input, w, b):
    cov = tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(tf.nn.bias_add(cov,b), name=name)

# generates a max pooling layer
def max_pool(name, l_input, k, s):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='VALID', name=name)

# generates a normalization layer
# tf.nn.lrn() = Local Response Normalization
# uses exactly the formula from the AlexNet paper.
#
# See:
# https://www.tensorflow.org/versions/master/api_docs/python/nn.html#local_response_normalization
#
# sqr_sum[a, b, c, d] =
#   sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
# output = input / (bias + alpha * sqr_sum) ** beta
#
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


# helper function to generate the AlexNet CNN architecture
def alex_net(_X, _dropout, n_classes, imagesize, img_channel):

    # prepare matrices for weights
    _weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, img_channel, 96])),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
        'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
        'wd1': tf.Variable(tf.random_normal([6*6*256, 4096])),
        'wd2': tf.Variable(tf.random_normal([4096, 4096])),
        'out': tf.Variable(tf.random_normal([4096, n_classes]))
    }
    
    # prepare vectors for biases
    _biases = {
        'bc1': tf.Variable(tf.random_normal([96])),
        'bc2': tf.Variable(tf.random_normal([256])),
        'bc3': tf.Variable(tf.random_normal([384])),
        'bc4': tf.Variable(tf.random_normal([384])),
        'bc5': tf.Variable(tf.random_normal([256])),
        'bd1': tf.Variable(tf.random_normal([4096])),
        'bd2': tf.Variable(tf.random_normal([4096])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # reshape input picture
    _X = tf.reshape(_X, shape=[-1, imagesize, imagesize, img_channel])


    # 1st convolution layer is a special case (4x4 stride, no padding)
    with tf.name_scope('Layer1'):

        conv1 = conv1st('conv1', _X, _weights['wc1'], _biases['bc1'])

        # normalization layer
        norm1 = norm('norm1', conv1, lsize=4)

        # max pooling layer
        pool1 = max_pool('pool1', norm1, k=3, s=2)

  

    
    with tf.name_scope('Layer2'):

        # convolution Layer
        conv2 = conv2d('conv2', pool1, _weights['wc2'], _biases['bc2'])


        # normalization layer
        norm2 = norm('norm2', conv2, lsize=4)

        # max pooling layer
        pool2 = max_pool('pool2', norm2, k=3, s=2)

 

    

    # more convolution layers
    with tf.name_scope('Layer3'):
        conv3 = conv2d('conv3', pool2, _weights['wc3'], _biases['bc3'])

    with tf.name_scope('Layer4'):
        conv4 = conv2d('conv4', conv3, _weights['wc4'], _biases['bc4'])

    with tf.name_scope('Layer5'):
        conv5 = conv2d('conv5', conv4, _weights['wc5'], _biases['bc5'])

        # max pooling layer
        pool3 = max_pool('pool3', conv5, k=3, s=2)


    # fully connected layer
    # reshape conv3 output to fit dense layer input
    with tf.name_scope('Layer6'):
        dense1 = tf.reshape(pool3, [-1, _weights['wd1'].get_shape().as_list()[0]])

        # relu activation function
        dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')

    with tf.name_scope('Layer7'):
        dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2')

    # output, class prediction
    with tf.name_scope('Layer8'):
        out = tf.matmul(dense2, _weights['out']) + _biases['out']

    return [out, _weights['wc1']]




# helper function to generate a small AlexNet
def small_alex_net(_X, _dropout, n_classes, imagesize, img_channel):

    # prepare matrices for weights
    _weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, img_channel, 10])),
        'wc2': tf.Variable(tf.random_normal([5, 5, 10, 15])),
        'wc3': tf.Variable(tf.random_normal([3, 3, 15, 20])),
        'wc4': tf.Variable(tf.random_normal([3, 3, 20, 25])),
        'wc5': tf.Variable(tf.random_normal([3, 3, 25, 30])),
        'wd1': tf.Variable(tf.random_normal([6*6*30, 40])),
        'wd2': tf.Variable(tf.random_normal([40, 40])),
        'out': tf.Variable(tf.random_normal([40, n_classes]))
    }
    
    # prepare vectors for biases
    _biases = {
        'bc1': tf.Variable(tf.random_normal([10])),
        'bc2': tf.Variable(tf.random_normal([15])),
        'bc3': tf.Variable(tf.random_normal([20])),
        'bc4': tf.Variable(tf.random_normal([25])),
        'bc5': tf.Variable(tf.random_normal([30])),
        'bd1': tf.Variable(tf.random_normal([40])),
        'bd2': tf.Variable(tf.random_normal([40])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # reshape input picture
    _X = tf.reshape(_X, shape=[-1, imagesize, imagesize, img_channel])

    with tf.name_scope('Layer1'):
        # 1st convolution layer is a special case (4x4 stride, no padding)
        conv1 = conv1st('conv1', _X, _weights['wc1'], _biases['bc1'])

        # normalization layer
        norm1 = norm('norm1', conv1, lsize=4)

        # max pooling layer
        pool1 = max_pool('pool1', norm1, k=3, s=2)

    
    with tf.name_scope('Layer2'):
        # convolution Layer
        conv2 = conv2d('conv2', pool1, _weights['wc2'], _biases['bc2'])

        # normalization layer
        norm2 = norm('norm2', conv2, lsize=4)

        # max pooling layer
        pool2 = max_pool('pool2', norm2, k=3, s=2)


    

    # more convolution layers
    with tf.name_scope('Layer3'):
        conv3 = conv2d('conv3', pool2, _weights['wc3'], _biases['bc3'])

    with tf.name_scope('Layer4'):
        conv4 = conv2d('conv4', conv3, _weights['wc4'], _biases['bc4'])

    with tf.name_scope('Layer5'):
        conv5 = conv2d('conv5', conv4, _weights['wc5'], _biases['bc5'])

        # max pooling layer
        pool3 = max_pool('pool3', conv5, k=3, s=2)


    # fully connected layer
    # reshape conv3 output to fit dense layer input
    with tf.name_scope('Layer6'):
        dense1 = tf.reshape(pool3, [-1, _weights['wd1'].get_shape().as_list()[0]])
        

        # relu activation function
        dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')
    

    with tf.name_scope('Layer7'):
        dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2')

    with tf.name_scope('Layer8'):
        # output, class prediction
        out = tf.matmul(dense2, _weights['out']) + _biases['out']

    return [out, _weights['wc1']]
