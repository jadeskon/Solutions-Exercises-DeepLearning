# Learning to classifiy digits using a MLP
# [variant with TB=TensorBoard visualization]
#
# Here we construct a multi-layer perceptron (MLP),
# i.e., a neural network using hidden layers,
# and then train it using the MNIST data
#
# ---
# by Prof. Dr. Juergen Brauer, www.juergenbrauer.org
 
import tensorflow as tf
 
# 1. we use the tensorflow.examples.tutorials.mnist class
#    in order to access the MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
print("\nmnist has type", type(mnist))
print("There are ", mnist.train.num_examples, " training examples available.")
print("There are ", mnist.test.num_examples, " test examples available.")
 
 
# 2. set learning parameters
learn_rate      = 0.05
nr_train_epochs = 5
batch_size      = 100
logs_path       = './logfiles'
 
 
# 3. configure network parameters
n_hidden_1 = 80  # nr of neurons in 1st hidden layer
n_hidden_2 = 20  # nr of neurons in 2nd hidden layer
n_input    = 784 # MNIST data input size:
                 # one input image has dimension 28x28 pixels,
                 # thus 784 input pixels
n_classes  =  10 # MNIST total classes (0-9 digits)
 
 
# 4. define TensorFlow input placeholders
#    A placeholder is simply a variable that we
#    will assign data to at a later date.
#
#    input x and output y will be 2D matrices
#    1st dimension of x is e.g. batch size,
#    2nd dimension of x is size of a single input image/vector
#
with tf.name_scope('Input'):
        x = tf.placeholder("float", [None, n_input])
with tf.name_scope('Teacher'):
        y = tf.placeholder("float", [None, n_classes])
#print("type of x is ", type(x))
#print("type of y is ", type(y))
 
 
# 5. helper function to create a 3 layer MLP:
#      input-layer -->
#       hidden layer #1 -->
#        hidden layer #2 -->
#         output layer
def multilayer_perceptron(x, weights, biases):
 
    # hidden layer #1 with RELU
    with tf.name_scope('Layer1'):
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            print("type of layer_1 is = ", type(layer_1))
     
    # hidden layer #2 with RELU
    with tf.name_scope('Layer2'):
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.relu(layer_2)
     
    # output layer with linear activation (no RELUs!)
    with tf.name_scope('OutputLayer'):
            out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
     
    # return the MLP model
    return out_layer
 
     
# 6. combine weights & biases of all layers in dictionaries
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
 
 
# 7. use helper function defined before to generate a MLP
with tf.name_scope('MyMLP'):
        my_mlp = multilayer_perceptron(x, weights, biases)
print("type of my_mlp is ", type(my_mlp))
 
 
# 8. define error function (generate an error op)
#    
#    1. see documentation for that TF function:
#       https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard7/tf.nn.softmax_cross_entropy_with_logits.md
#    
#    2. "If you want to do optimization to minimize the cross entropy,
#        AND you're softmaxing after your last layer, you should use
#        tf.nn.softmax_cross_entropy_with_logits"
#        see http://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with 
#
#    3. "The logits are the unnormalized log probabilities"
#       http://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with 
#
#
# Uses the cross-entropy error between the actual output values
# (my_mlp, output values are first softmax normalized) and the
# desired output values
with tf.name_scope('ErrorFunction'):
        error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(my_mlp, y))

 
 
# 9. define optimizer

# ca. 93% accuracy for settings
with tf.name_scope('GradientDescentOptimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(error)

# < 90% accuracy for settings
#optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(error)
#optimizer = tf.train.MomentumOptimizer(learning_rate=0.3, momentum=0.2).minimize(error)
 
 
# 10. initialize all variables defined in the model and launch the graph
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

####################### TensorBoard stuff #########################

# create summary op to monitor loss function op
tf.scalar_summary("TrainingError", error)

 
# merge all summaries into a single op to make collection of log data
# more convenient in the following
merged_summary_op = tf.merge_all_summaries()
 
 
# create summary_writer object to write logs to TensorBoard
summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

####################### TensorBoard stuff #########################
 
 
# 11. the actual training happens here:
for epoch in range(nr_train_epochs):
 
    # reset epoch error
    epoch_error = 0.0
 
    # compute how many batches we will have to process
    nr_batches_to_process = int(mnist.train.num_examples/batch_size)
     
    # loop over all batches to process
    for i in range(nr_batches_to_process):
     
        # get next training batch input matrix and batch label vector
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        #print("type of batch_x is ", type(batch_x))
        #print("shape of batch_x is ", batch_x.shape)
         
        # Run optimization op (backprop) and error op
        _, current_error, summary = sess.run([optimizer, error, merged_summary_op],
                                    feed_dict={x: batch_x, y: batch_y})   
                                                       
        # compute total error of all batches in this epoch
        epoch_error += current_error
         
    # display epoch nr and accumated error
    # for the selected batches in this epoch
    print("Epoch:", '%03d' % epoch, ", epoch error=", "{:.3f}".format(epoch_error))

    summary_writer.add_summary(summary, epoch)
     
print("Optimization Finished!")
 

 
# 12. calculate accuracy of the learned model
#     on the test dataset
correct_prediction = tf.equal(tf.argmax(my_mlp, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 
print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

