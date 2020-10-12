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

    def getLocalizationMap(self, image, c = 'None'):
        gradModel = Model(inputs = self.model.inputs, outputs = [self.model.get_layer(self.layer).output, self.model.output])
        with tf.GradientTape() as t:
            (featureMaps, predictions) = gradModel(image)
            if c == 'None':
                c = np.argmax(predictions[0])
            c = int(c)
            score = predictions[:,c]
                
        grads = t.gradient(score, featureMaps)
        alpha_c_k = tf.reduce_mean(grads, axis=(1,2))
        locMap = np.maximum(tf.reduce_sum(tf.multiply(alpha_c_k, featureMaps), axis=3).numpy()[0], 0)
        # 14 x 14 Map
        return locMap, c

    def getHeatmap(self, locMap, image):
        locMapResized = cv2.resize(locMap, (224, 224))
        heatmap = locMapResized / np.max(locMapResized)
        colorMap = cv2.applyColorMap(np.uint8(heatmap*255), cv2.COLORMAP_JET)
        cm_rgb = cv2.cvtColor(colorMap, cv2.COLOR_BGR2RGB)
        image = image[0, :]
        image -= np.min(image)
        image = np.minimum(image, 255)
        overLayed = cm_rgb + image
        overLayed = 255 * overLayed / np.max(overLayed)
        return overLayed

def preProcessImage(imagePath):

    image = I.load_img(imagePath, target_size=(224,224))
    image = I.img_to_array(image)
    image = np.reshape(image,(1, 224,224,3))
    image = preprocess_input(image)
    return image

def parseArgs():
    parser = argparse.ArgumentParser(description='GradCAM')
    parser.add_argument('--imagePath', default='None', type=str)
    parser.add_argument('--imageClass', default='None', type=str)
    parser.add_argument('--dataPath', default='images/', type=str)
    parser.add_argument('--resultsPath', default='heatmaps/heatmap_', type=str)
    parser.add_argument('--layer', default='block5_conv3', type=str)

    return parser.parse_args()

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    args = parseArgs()

    model = VGG16(weights='imagenet')

    gradCAM = GradCAM(model, args.layer)

    labelsMap = pickle.load(open('labelsMap.p', 'rb'))
    classMap = pickle.load(open('classMap.p', 'rb'))
    resultsFile = open('./classRelation.txt', 'w')

    if not args.imagePath == 'None':
        path = args.imagePath
        image = preProcessImage(path)
        c = classMap.get(args.imageClass, args.imageClass)
        locMap, c = gradCAM.getLocalizationMap(image, c)
        heatMap = gradCAM.getHeatmap(locMap, image)
        name = os.path.split(path)[-1]
        plt.imsave(args.resultsPath + args.layer + '_' + str(c) + '_' + name, np.uint8(heatMap))

    else:

        for root, dirs, files in os.walk(args.dataPath):
            for name in files:
                path = os.path.join(root, name)
                image = preProcessImage(path)
                locMap, c = gradCAM.getLocalizationMap(image)
                heatMap = gradCAM.getHeatmap(locMap, image)
                
                resultsFile.write(path + ' ---> ' + labelsMap[c] + '\n')
                plt.imsave(args.resultsPath + args.layer + '_' + name, np.uint8(heatMap))

    resultsFile.close()

if __name__ == "__main__":
    main()

