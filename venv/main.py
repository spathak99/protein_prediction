from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import gzip
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

learning_rate = 1e-3
batch_size = 100
data = np.float32(np.reshape(np.load('data/cullpdb+profile_6133.npy.gz'), (6133, 700, 57, 1)))
dataindex = range(22)
labelindex = range(22, 32)
training_labels = data[:5600, :, labelindex]
training_data = data[:5600, :, dataindex]
test_data = data[5605:5877, :, dataindex]
test_label = data[5605:5877, :, labelindex]
training_data.shape = [5600,700,22]
print(test_label.shape, " ", test_data.shape)
test_label = tf.reshape(test_label,[272,700,10])
test_data = tf.reshape(test_data,[272,700,22])


#x = tf.placeholder(tf.float32, shape=[None,training_data.shape[1],training_data.shape[2]])
#y = tf.placeholder(tf.float32, shape=[None,training_labels.shape[1],training_labels.shape[2]])

output_dim=training_labels.shape[2]
num_nodes = 100

wa = tfe.Variable(tf.random_normal([data.shape[1], num_nodes, num_nodes]))
ba = tfe.Variable(tf.random_normal([data.shape[1], 1, wa.shape[1].value]))
wx = tfe.Variable(tf.random_normal([data.shape[1], num_nodes, training_data.shape[2]]))
bx = tfe.Variable(tf.random_normal([data.shape[1], 1, wx.shape[1].value]))
wy = tfe.Variable(tf.random_normal([data.shape[1], output_dim, num_nodes]))
by = tfe.Variable(tf.random_normal([data.shape[1], 1, output_dim]))

def train(x, y):
    preds = forward_prop(x)
   # print(preds.shape)
    print("preds", preds.shape)
    cost = tf.reduce_mean(
        tf.reduce_sum(
            tf.subtract(
                tf.multiply(
                    tf.cast(tf.multiply(-1,y),tf.float32),
                    tf.log(preds)),
                tf.multiply(
                    tf.subtract(1, y),
                    tf.log(
                        tf.subtract(1, preds)))),
            axis=1),
        axis=0)
    print(cost)
    return cost

def timestep(data, a, stepindex=0, first=False):
    if(first):
        a = tfe.Variable(tf.zeros([data.shape[0], num_nodes]))

    a1 = [tf.sigmoid(
        tf.add(
            tf.matmul(a, tf.transpose(wa[stepindex])),
            ba[stepindex])),
        tf.sigmoid(
            tf.add(
                tf.matmul(data,
                          tf.transpose(wx[stepindex])),
                bx[stepindex]))]

    conc = tf.add(a1[0],a1[1])
    #print(conc.shape)

    yh = tf.nn.softmax(tf.add((tf.matmul(conc, tf.transpose(wy[stepindex]))), by[stepindex]))
    #print("y HAT: ",yh.shape)
    return conc, yh


def forward_prop(data):
    preds = []#tfe.Variable([], dtype=tf.float32) #tfe.Variable(dtype=tf.float32, graph=None)
    first = True
    curr_a,curr_y = timestep(tf.transpose(data,[1,0,2])[0],None,first=True)
    preds.append(curr_y)
    #return tf.map_fn(lambda x: timestep(x),tf.transpose(data,[1,0,2])[1:])
    for i in tf.transpose(data,[1,0,2]):
        if first:
            first = False
        else:
            curr_a,curr_y = timestep(i,curr_a)
            preds.append(curr_y)
    preds = tf.convert_to_tensor(preds, dtype=tf.float32)
    preds = tf.transpose(preds, [1, 0, 2])
    return preds




optimizer = tf.train.AdamOptimizer(learning_rate = 1e-2)
for i in range(0, training_labels.shape[0], batch_size):
    train_batch = training_data[i:i + batch_size, :, :]
    label_batch = training_labels[i:i + batch_size, :, :]
    optimizer.minimize(lambda: train(tf.reshape(train_batch,
            [train_batch.shape[0],train_batch.shape[1],train_batch.shape[2]]),
            tf.reshape(label_batch,
            [label_batch.shape[0],label_batch.shape[1],
           label_batch.shape[2]])))
preds = forward_prop(test_data)
c = 0.0
t = 0.0
for x in range(preds.shape[0].value):
    for y in range(preds.shape[1].value):
        if(np.argmax(preds[x,y,:]) == np.argmax(test_label[x,y,:])):
            t+=1
    t=float(t)/float(preds.shape[1].value)
    c+=t
acc = float(c)/float(preds.shape[0].value)
print(acc)


