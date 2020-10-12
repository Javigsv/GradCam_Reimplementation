from argparse import HelpFormatter
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import os
import pickle
from numpy.lib.function_base import insert

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import Model
import tensorflow.keras.preprocessing.image as I

class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer

    def getLocalizationMap(self, image):
        gradModel = Model(inputs = self.model.inputs, outputs = [self.model.get_layer(self.layer).output, self.model.output])
        with tf.GradientTape() as t:
            (featureMaps, predictions) = gradModel(image)
            c = np.argmax(predictions[0])
            score = predictions[:,c]
        grads = t.gradient(score, featureMaps)
        alpha_c_k = tf.reduce_mean(grads, axis=(1,2))
        locMap = np.maximum(tf.reduce_sum(tf.multiply(alpha_c_k, featureMaps), axis=3).numpy()[0], 0)
        # 14 x 14 Map
        return locMap, c

    def getHeatmap(self, locMap, image):
        locMapResized = cv2.resize(locMap, (224, 224))
        heatmap = locMapResized / np.max(locMapResized)
        colorMap = cv2.applyColorMap(np.uint8(heatmap*255), cv2.COLORMAP_HOT)
        image = image[0, :]
        image -= np.min(image)
        image = np.minimum(image, 255)
        overLayed = colorMap + image
        overLayed = 255 * overLayed / np.max(overLayed)
        return overLayed

def preProcessImages(imagePath):

    image = I.load_img(imagePath, target_size=(224,224))
    image = I.img_to_array(image)
    image = np.reshape(image,(1, 224,224,3))
    image = preprocess_input(image)
    return image

def parseArgs():
    parser = argparse.ArgumentParser(description='GradCAM')
    parser.add_argument('--imagePath', default='images/', type=str)
    parser.add_argument('--dataPath', default='images/', type=str)
    parser.add_argument('--resultsPath', default='heatmaps/', type=str)

    return parser.parse_args()

def main():
    args = parseArgs()

    for root, dirs, files in os.walk(args.dataPath):
        for name in files:
            path = os.path.join(root, name)
            image = preProcessImage(path)
            model = VGG16(weights='imagenet')
            gradCAM = GradCAM(model, 'block5_conv3')
            locMap, c = gradCAM.getLocalizationMap(image)
            heatMap = gradCAM.getHeatmap(locMap, image)
            
            labelsMap = pickle.load(open('labelsMap.p', 'rb'))
            label = '_' + labelsMap[c] + '.'
            name = label.join(name.split('.'))
            plt.imsave(args.resultsPath + name, np.uint8(heatMap))


if __name__ == "__main__":
    main()

