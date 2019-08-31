
# coding: utf-8

import tensorflow as tf
from tf_version import load_checkpoint,Net
from PIL import Image
import numpy as np

def eval_(net,path):
    inputs = tf.placeholder(tf.float32,[None,128,128,3])
    output, class_ret = net.res(inputs)
    class_output_val = tf.nn.softmax(class_ret)
    
    saver = tf.train.Saver()
    Im = np.array(Image.open('face-0.bmp')).reshape(-1,128,128,3)
    with tf.Session() as sess:
        load_checkpoint(saver,sess,path)
        class_output_= sess.run(class_output_val,feed_dict={inputs:Im})
        print(class_output_.shape)
        score = class_output_[0][1]
    return score


net = Net()
path = './model'

eval_(net,path)

