#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import sys
import cv2
import numpy as np

model_full_path = 'model/retrained_graph.pd'

labels_full_path = 'model/retrained_labels.txt'


def get_answer_from_cv2_image(cv2_image):
    # # Read in the image_data
    # image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    image_copy = cv2_image.copy()
    # Format for the Mul:0 Tensor
    image_copy = cv2.resize(image_copy, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
    # Numpy array
    np_image_data = np.asarray(image_copy)
    np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
    tensor_image_data = np.expand_dims(np_image_data, axis=0)

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile(labels_full_path)]

    # Unpersists graph from file
    with tf.gfile.FastGFile(model_full_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        # predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        predictions = sess.run(softmax_tensor, {'Mul:0': tensor_image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))

    # get most likely classification
    answer = label_lines[top_k[0]]
    return answer


def get_answer_from_file(image_path):
    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile(labels_full_path)]

    # Unpersists graph from file
    with tf.gfile.FastGFile(model_full_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))

    # get most likely classification
    answer = label_lines[top_k[0]]
    return answer


if __name__ == '__main__':
    # get image to classify
    test_image_path = sys.argv[1]
    image = cv2.imread(test_image_path)
    ans = get_answer_from_cv2_image(image)
    print("result? : " + ans)
    if ans == "text":
        print("yes!")
    elif ans == "not text":
        print("no! ")
