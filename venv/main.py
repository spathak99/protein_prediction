from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import gzip
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import math

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

wa = tfe.Variable(tf.random_normal([num_nodes, num_nodes]))
ba = tfe.Variable(tf.random_normal([1, wa.shape[0].value]))
wx = tfe.Variable(tf.random_normal([num_nodes, training_data.shape[2]]))
bx = tfe.Variable(tf.random_normal([1, wx.shape[0].value]))
wy = tfe.Variable(tf.random_normal([output_dim, num_nodes]))
by = tfe.Variable(tf.random_normal([1, output_dim]))

#incx = tfe.Variable(tf.random_normal([1, training_data.shape[2]]))
#inca = tfe.Variable(tf.random_normal([1, num_nodes]))
#bi = tfe.Variable(tf.random_normal([1]))

fcx = tfe.Variable(tf.random_normal([1, training_data.shape[2]]))
fca = tfe.Variable(tf.random_normal([1, num_nodes]))
bf = tfe.Variable(tf.random_normal([1]))


wcx = tfe.Variable(tf.random_normal([num_nodes, training_data.shape[2]]))
wca = wa = tfe.Variable(tf.random_normal([num_nodes, num_nodes]))
bc = tfe.Variable(tf.random_normal([1,num_nodes]))

wox = tfe.Variable(tf.random_normal([1, training_data.shape[2]]))
woa = tfe.Variable(tf.random_normal([1, num_nodes]))
bo = tfe.Variable(tf.random_normal([1]))

def lstm_cell(data, a, c, first=False):
    if(first):
        c = tfe.Variable(tf.zeros([data.shape[0], num_nodes]))

    #in_gate = tf.sigmoid(tf.add(tf.add(tf.matmul(data, tf.transpose(incx)), tf.matmul(a, tf.transpose(inca))), bi))
    forget_gate = tf.sigmoid(tf.add(tf.add(tf.matmul(data, tf.transpose(fcx)), tf.matmul(a, tf.transpose(fca))), bf))
    #print(forget_gate)
    #print("in gate", in_gate, " forget gate", forget_gate)

    nextc = tf.tanh(tf.add(tf.add(tf.matmul(a,tf.transpose(wca)), tf.matmul(data,tf.transpose(wcx))), bc))

    c = tf.add(tf.multiply(forget_gate, c), tf.multiply(tf.subtract(1, forget_gate), nextc))

    out_gate = tf.sigmoid(tf.add(tf.add(tf.matmul(data, tf.transpose(wox)), tf.matmul(a, tf.transpose(woa))), bo))

    return c, out_gate

def train(x, y):
    preds = forward_prop(x)
   # print(preds.shape)
    #print("preds", preds.shape)
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

def timestep(data, a, c, first=False):
    out_gate = 0
    if(first):
        a = tfe.Variable(tf.zeros([data.shape[0], num_nodes]))
        c, out_gate = lstm_cell(data, a, None, first = True)
    else:
        c, out_gate = lstm_cell(data, a, c)

    #a1 = [tf.sigmoid(tf.add(tf.matmul(a, tf.transpose(wa)),ba)),tf.sigmoid(tf.add(tf.matmul(data,tf.transpose(wx)),bx))]
    a1 = [tf.sigmoid(
        tf.add(
            tf.matmul(a, tf.transpose(wa)),
            ba)),
        tf.sigmoid(
            tf.add(
                tf.matmul(data,
                          tf.transpose(wx)),
                bx))]

    conc = (tf.add(a1[0],a1[1]))
    #conc = tf.add(tf.multiply(out_gate, c), tf.multiply(tf.subtract(1, out_gate), conc))

    #print("out gate", out_gate)

    yh = tf.nn.softmax(tf.add((tf.matmul(conc, tf.transpose(wy))), by))
    #print("y HAT: ",yh.shape)
    return conc, yh, c


def forward_prop(data):
    preds = []#tfe.Variable([], dtype=tf.float32) #tfe.Variable(dtype=tf.float32, graph=None)
    first = True
    curr_a,curr_y,curr_c = timestep(tf.transpose(data,[1,0,2])[0],None, None, first=True)
    preds.append(curr_y)
    #return tf.map_fn(lambda x: timestep(x),tf.transpose(data,[1,0,2])[1:])
    for i in tf.transpose(data,[1,0,2]):
        if first:
            first = False
        else:
            curr_a,curr_y,curr_c = timestep(i, curr_a, curr_c)
            preds.append(curr_y)
    preds = tf.convert_to_tensor(preds, dtype=tf.float32)
    preds = tf.transpose(preds, [1, 0, 2])
    #print(preds)
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


