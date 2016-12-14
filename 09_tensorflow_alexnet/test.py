# file: test.py
#
# Restores a learned AlexNet model and
# uses all the test images from the pics_test folder
# in order to compute the classification accuracy of
# the learned model.
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

from importData import Dataset
testing = Dataset('experiment01/pics_test', '.jpeg')

import tensorflow as tf
import numpy as np

# Parameters
batch_size = 1

ckpt = tf.train.get_checkpoint_state("save")
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

pred = tf.get_collection("pred")[0]
x = tf.get_collection("x")[0]
keep_prob = tf.get_collection("keep_prob")[0]

sess = tf.Session()
saver.restore(sess, ckpt.model_checkpoint_path)

# test
step_test = 1
correct=0
while step_test * batch_size < len(testing):
    testing_ys, testing_xs = testing.nextBatch(batch_size)
    predict = sess.run(pred, feed_dict={x: testing_xs, keep_prob: 1.})
    #print("\nImage test: ", step_test)
    #print("Testing label:", testing.label2category[np.argmax(testing_ys, 1)[0]])
    #print("Testing predict:", testing.label2category[np.argmax(predict, 1)[0]])

    groundtruth_label = np.argmax(testing_ys, 1)[0]
    predicted_label = np.argmax(predict, 1)[0]

    if predicted_label == groundtruth_label:
        correct+=1
    step_test += 1
print("Classified", correct, "images correct of", step_test,"in total.")
print("Classification rate = ", correct/step_test)

