from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import gzip
import tensorflow as tf

learning_rate = 1e-3
batch_size = 100
data = np.float32(np.reshape(np.load('data/cullpdb+profile_6133.npy.gz'), (6133, 700, 57, 1)))
dataindex = range(22)
labelindex = range(22, 32)
training_labels = data[:5600, :, labelindex]
training_data = data[:5600, :, dataindex]
training_data.shape = [5600,700,22]


x = tf.placeholder(tf.float32, shape=[batch_size,training_data.shape[1],training_data.shape[2]])
y = tf.placeholder(tf.float32, shape=[batch_size,training_labels.shape[1],training_labels.shape[2]])

def train():
    preds = forward_prop(x)
    preds = tf.transpose(preds,[1,0,2])
    cost = tf.reduce_mean(
        tf.reduce_sum(
            tf.subtract(
                tf.multiply(
                    tf.multiply(-1,y),
                    tf.log(preds)),
                tf.multiply(
                    tf.subtract(1, y),
                    tf.log(
                        tf.subtract(1, preds)))),
            axis=1),
        axis=0)
    print(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    return optimizer, preds

def timestep(data, output_dim=training_labels.shape[2], a = tf.Variable(tf.random_normal([batch_size,training_labels. shape[2]]))):

    wa = tf.Variable(tf.random_normal([output_dim, a.shape[1].value]))
    ba = tf.Variable(tf.random_normal([wa.shape[0].value,1]))
    wx = tf.Variable(tf.random_normal([output_dim,data.shape[1].value]))
    bx = tf.Variable(tf.random_normal([wx.shape[0].value,1]))
    wy = tf.Variable(tf.random_normal([data.shape[0].value,data.shape[0].value]))
    by = tf.Variable(tf.random_normal([wy.shape[0].value, 1]))
    a1 = [tf.sigmoid(
        tf.add(
            tf.matmul(wa,
                      tf.transpose(a)),
            ba)),
        tf.sigmoid(
            tf.add(
                tf.matmul(wx,
                          tf.transpose(data)),
                bx))]

    conc = tf.add(a1[0],a1[1])
    print(conc.shape)

    yh = tf.nn.softmax(tf.add(tf.matmul(wy, tf.transpose(conc)), by))
    print(yh.shape)
    return tf.transpose(conc), yh


def forward_prop(data):
    preds = []
    first = True
    a0,y0 = timestep(tf.transpose(data,[1,0,2])[0])
    preds = tf.map_fn(lambda x: timestep(x),tf.transpose(data,[1,0,2])[1:])
    for i in tf.transpose(data,[1,0,2]):
        if first:
            first = False
        else:
            a,y = timestep(i,a=a0)
            preds.append(y)
    return preds

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(0, training_labels.shape[0], batch_size):
        opt, preds = sess.run(train(), feed_dict={x:training_data[i:i+batch_size],y:training_labels[i,i+batch_size]})






"""
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
    #m_result3 = tf.reshape(m_result3,[5877,700,8])
    return {"result3":m_result3,
            "result2":m_result2,
            "result1":m_result1,
            "dense1": dense1,
            "dense2": dense2,
            "dense3": dense3}


def back_prop(data):
    lr = tf.Variable(0.15)
    data = forward_prop(data)
    res = data["result3"]
    train_y = tf.reshape(y[:, :, :, 0], res.shape)
    res = tf.square(tf.subtract(train_y,res))
    res = tf.divide(res,tf.multiply(2, data["result3"].shape[0]))
    res = tf.reduce_mean(res,axis=0)
    #res = tf.reduce_sum(res,axis=1)
    print("cost: ",res)
    res = tf.reshape(res,[res.shape[0],1])

back_prop(x)
"""
