from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import gzip
import tensorflow as tf
import keras
from keras.models import  Sequential
from keras.layers import TimeDistributed


def main():
    data = np.reshape(np.load('data/cullpdb+profile_6133.npy.gz'),(6133,700,57))
    dataindex = range(21)
    labelindex = range(22,30)
    ind = [30]
    training_data = data[:5600,:,dataindex]
    print(training_data)
    training_labels = data[:5600,:,labelindex]
    val_data = data[5600:5877,:,dataindex]
    val_labels = data[5600:5877,:,labelindex]
    training_data = np.concatenate((training_data, val_data), axis=0)
    training_labels = np.concatenate((training_labels, val_labels), axis=0)
    test_data = data[5877:,:,dataindex]
    test_label = data[5877:,:,labelindex]
    classifier = tf.estimator.Estimator(model_fn=model, model_dir="\model")
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": training_data},
        y=training_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": val_data},
        y=val_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)





