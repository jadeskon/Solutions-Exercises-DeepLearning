# file: train_alexnet.py
#
# AlexNet implementation in TensorFlow
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
#
# Store your training images in a folder experiment01/pics_train/
# e.g.
#    experiment01/pics_train/dogs
#    experiment01/pics_train/cows
#
# Store your test images in a folder experiment01/pics_test/
# e.g.
#    experiment01/pics_test/dogs
#    experiment01/pics_test/cows
#
# ---
# Prof. Dr. Juergen Brauer, www.juergenbrauer.org

from importData import Dataset
import inference
import tensorflow as tf


# 1. create a training and testing Dataset object that stores
#    the training / testing images
training = Dataset('experiment01/pics_train', '.jpeg')
testing  = Dataset('experiment01/pics_test',  '.jpeg')


# 2. set training parameters
learn_rate = 0.001
decay_rate = 0.1
batch_size = 32
display_step = 1
nr_mini_batches_to_train = 10000
save_filename = 'save/model.ckpt'
logs_path     = './logfiles'

n_classes = training.num_labels
dropout = 0.8 # dropout (probability to keep units)
imagesize = 227
img_channel = 3

with tf.name_scope('Input'):
   x = tf.placeholder(tf.float32, [None, imagesize, imagesize, img_channel])
with tf.name_scope('Teacher'):
   y = tf.placeholder(tf.float32, [None, n_classes])
with tf.name_scope('KeepProb'):
   keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


##################
# generate AlexNet
##################
with tf.name_scope('MyCNN'):
   #[pred, filter1st] = inference.alex_net(x, keep_prob, n_classes, imagesize, img_channel)
   [pred, filter1st] = inference.small_alex_net(x, keep_prob, n_classes, imagesize, img_channel)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(learn_rate, global_step, 1000, decay_rate, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost, global_step=global_step)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
saver = tf.train.Saver()
tf.add_to_collection("x", x)
tf.add_to_collection("y", y)
tf.add_to_collection("keep_prob", keep_prob)
tf.add_to_collection("pred", pred)
tf.add_to_collection("accuracy", accuracy)

print("\n\n")
print("----------------------------------------")
print("I am ready to start the training...")
print("So I will train an AlexNet, starting with a learn rate of", learn_rate)
print("I will exponentially decay the learn rate with a decay rate of ", decay_rate)
print("I will train ", nr_mini_batches_to_train, "mini batches of ", batch_size, "images")
print("Your input images will be resized to ", imagesize, "x", imagesize, "pixels")
print("----------------------------------------")


with tf.Session() as sess:
    sess.run(init)


    ####################### TensorBoard stuff #########################

    # create summary ops to monitor some scalar values
    tf.scalar_summary("Accuracy", accuracy)
    tf.scalar_summary("LearningRate", lr)

    # create a summary op for visualizing the filters of the 1st layer
    #filter_summary = tf.image_summary("weightvisus", filter1st[:,:,:,1])

     
    # merge all summaries into a single op to make collection of log data
    # more convenient in the following
    merged_summary_op = tf.merge_all_summaries()
     
     
    # create summary_writer object to write logs to TensorBoard
    summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

    ####################### TensorBoard stuff #########################


    ###############
    # 1. training
    ###############
    step = 1
    while step < nr_mini_batches_to_train:
        batch_ys, batch_xs = training.nextBatch(batch_size)

        _, summary = sess.run([optimizer, merged_summary_op],
                              feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})

        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            rate = sess.run(lr)
            print("learn rate:" + str(rate) + " mini batch:" + str(step) + ", minibatch loss= " + "{:.6f}".format(loss) + ", batch accuracy= " + "{:.5f}".format(acc))
            summary_writer.add_summary(summary, step)

        if step % 100 == 0:              
            print("I will save the model to ", save_filename)
            saver.save(sess, save_filename, global_step=step*batch_size)
        step += 1
    print("Training finished!")




    #############################
    # 2. test model just trained
    #############################
    print("\n")
    nr_test_images = len(testing)
    print("Testing your model with", nr_test_images, "test images")

    sum_accuracies = 0
    batch_nr = 0
    while batch_nr * batch_size < len(testing):
        testing_ys, testing_xs = testing.nextBatch(batch_size)
        batch_accuracy = sess.run(accuracy, feed_dict={x: testing_xs, y: testing_ys, keep_prob: 1.})
        print("\taccuracy for batch", batch_nr, ":", batch_accuracy)
        sum_accuracies += batch_accuracy
        batch_nr +=1 
    print("Average accuracy:", sum_accuracies/batch_nr)



print("\n")
print("AlexNet model training and model accuracy testing finished.")
