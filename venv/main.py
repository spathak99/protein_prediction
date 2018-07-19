from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import gzip
import tensorflow as tf
import keras



data = np.float32(np.reshape(np.load('data/cullpdb+profile_6133.npy.gz'), (6133, 700, 57, 1)))
dataindex = range(21)
labelindex = range(22, 30)
ind = [30]
training_data = data[:5600, :, dataindex]
#print(training_data)
training_labels = data[:5600, :, labelindex]
val_data = data[5600:5877, :, dataindex]
val_labels = data[5600:5877, :, labelindex]
training_data = np.concatenate((training_data, val_data), axis=0)
training_labels = np.concatenate((training_labels, val_labels), axis=0)
x = tf.placeholder(tf.float32, shape = [training_data.shape[0],training_data.shape[1],training_data.shape[2],1])
y = tf.placeholder(tf.float32, shape = [training_labels.shape[0],training_labels.shape[1],training_labels.shape[2],1])
test_data = data[5877:, :, dataindex]
test_label = data[5877:, :, labelindex]

conv1 = {"weights": tf.Variable(tf.random_normal([5,5,1,50])),
          "biases": tf.Variable(tf.random_normal([50]))}

conv2 = {"weights": tf.Variable(tf.random_normal([10,10,50,60])),
          "biases": tf.Variable(tf.random_normal([60]))}

conv3 = {"weights": tf.Variable(tf.random_normal([20,20,60,80])),
          "biases": tf.Variable(tf.random_normal([80]))}




def forward_prop(data):
    calc1 = tf.add(tf.nn.conv2d(data,conv1["weights"],strides=[1,1,1,1],padding="SAME"),conv1["biases"])
    pool1 = tf.nn.max_pool(calc1,ksize=[1,4,4,1],strides=[1,2,2,1],padding="SAME")
    calc2 = tf.add(tf.nn.conv2d(pool1,conv2["weights"],strides=[1,1,1,1],padding="SAME"),conv2["biases"])
    pool2 = tf.nn.max_pool(calc2,ksize=[1,4,4,1],strides=[1,2,2,1],padding="SAME")
    calc3 = tf.add(tf.nn.conv2d(pool2,conv3["weights"],strides=[1,1,1,1],padding="SAME"),conv3["biases"])
    calc3 = tf.reshape(calc3,tf.TensorShape([calc3.shape[0].value,calc3.shape[1].value * calc3.shape[2].value * calc3.shape[3].value]))

    dense1 = {"weights": tf.Variable(tf.random_normal([50, calc3.shape[1].value])),
              "biases": tf.Variable(tf.random_normal([50]))}

    dense2 = {"weights": tf.Variable(tf.random_normal([50, dense1["weights"].shape[0].value])),
              "biases": tf.Variable(tf.random_normal([50]))}

    dense3 = {"weights": tf.Variable(tf.random_normal([5600, dense2["weights"].shape[0].value])),
              "biases": tf.Variable(tf.random_normal([5600]))}

    m_result1 = tf.tanh(tf.add(tf.matmul(calc3, tf.transpose(dense1["weights"])),tf.reshape(dense1["biases"],[1,50])))
    m_result2 = tf.tanh(tf.add(tf.matmul(m_result1, tf.transpose(dense2["weights"])),tf.reshape(dense2["biases"],[1,50])))
    m_result3 = tf.sigmoid(tf.add(tf.matmul(m_result2, tf.transpose(dense3["weights"])),tf.reshape(dense3["biases"],[1,5600])))
    m_result3 = tf.reshape(m_result3,[5877,700,8])
    return {"result3":m_result3, "result2":m_result2,"result1":m_result1, "dense1": dense1, "dense2": dense2, "dense3": dense3}


def back_prop(data):
    lr = tf.Variable(0.15)
    data = forward_prop(data)
    res = data["result3"]
    res = tf.square(tf.subtract(y[:,:,:,0],res))
    res = tf.reduce_sum(res,axis=1)
    res = tf.reduce_sum(res,axis=1)
    res = tf.reshape(res,[res.shape[0],1])
    dz = tf.multiply(res,lr)
    data["dense3"]["weights"] = data["dense3"]["weights"] + tf.transpose(tf.matmul(tf.transpose(data["result2"]), dz))


back_prop(x)
