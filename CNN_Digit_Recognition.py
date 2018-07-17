# -*- coding: utf-8 -*-
"""
Created on Sun Mar    4 13:25:23 2018

@author: cyriac.azefack
"""

import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 50, 50, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs = input_layer,
            filters = 32,
            kernel_size = [5, 5],
            padding = "same",
            activation = tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
                    inputs = conv1,
                    pool_size = [2, 2],
                    strides = 2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
            inputs = pool1,
            filters = 64,
            kernel_size = [5, 5],
            padding = "same",
            activation = tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(
                    inputs = conv2,
                    pool_size = [2, 2],
                    strides = 2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 12 * 12 * 64])
    dense = tf.layers.dense(
                    inputs = pool2_flat,
                    units = 1024,
                    activation = tf.nn.relu)

    dropout = tf.layers.dropout(
            inputs=dense,
            rate = 0.4,
            training = mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(
                    inputs = dropout,
                    units=10)

    predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs={
            'predict' : tf.estimator.export.PredictOutput(predictions)
        })

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, 
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, 
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, 
                                      eval_metric_ops=eval_metric_ops)


def serving_input_receiver_fn():
    inputs = {
        'x': tf.placeholder(tf.float32, [None, 2500]),
    }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def main():
    # Load training and eval data
    with open('data.pickle', 'rb') as file:
            data_dict = pickle.load(file)

    features = data_dict['features']
    labels = data_dict['labels']

    features = features.astype(np.float32)
    train_features, test_features, train_labels, test_labels = train_test_split(
                    features, labels, test_size = 0.2)

    # Create the Estimator
    classifier_model = tf.estimator.Estimator(
                    model_fn=cnn_model_fn, model_dir="./classifier_model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": train_features},
                    y=train_labels,
                    batch_size=500,
                    num_epochs=None,
                    shuffle=True)

    classifier_model.train(
            input_fn=train_input_fn,
            max_steps=5000,
            hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": test_features},
                    y=test_labels,
                    num_epochs=1,
                    shuffle=False)

    eval_results = classifier_model.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    # SAVE THE MODEL
    full_model_dir = classifier_model.export_savedmodel(export_dir_base="./classifier_model",
                                                        serving_input_receiver_fn=serving_input_receiver_fn)
    print('Model Saved into : {}'.format(full_model_dir))

    # Test prediction
    feat = test_features[6]
    plt.imshow(feat.reshape((50, 50)))
    plt.show()

    test = np.array([feat], dtype=np.float32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": test},
                    num_epochs=1,
                    shuffle=False)
    predictions = list(classifier_model.predict(input_fn=predict_input_fn))
    predicted_classes = [p["classes"] for p in predictions]
    print("New Samples, Class Predictions:{}\n".format(predicted_classes))
    

#

if __name__ == '__main__':
        main()