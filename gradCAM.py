from argparse import HelpFormatter
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import os
import pickle
from numpy.lib.function_base import average, insert
from tqdm import tqdm

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
    
    def evaluate(self, locMap, imagePath, c, tolerance):
        image = I.load_img(imagePath,target_size=(224,224))
        image = I.img_to_array(image)
        locMapResized = cv2.resize(locMap, (224, 224))
        heatmap = locMapResized / np.max(locMapResized)
        _, thresh_img = cv2.threshold(np.uint8(heatmap*255), tolerance, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        black_img = np.zeros(image.shape).astype(image.dtype)
        color = [1, 1, 1]
        cv2.fillPoly(black_img, contours, color)
        segmented_img = image*black_img
        segmented_img = np.reshape(segmented_img,(1, 224,224,3))
        segmented_img = preprocess_input(segmented_img)
        original_img = np.reshape(image,(1, 224,224,3))
        original_img = preprocess_input(original_img)
        Y = self.model(original_img)[0,c].numpy()
        O = self.model(segmented_img)[0,c].numpy()
        drop = np.maximum(0,Y-O)/Y
        inc = 0
        if Y < O:
            inc = 1
        return drop, inc

class GradCAMPlusPlus(GradCAM):
    def __init__(self, model, layer):
        super().__init__(model, layer)

    def getLocalizationMap(self, image, c = 'None'):
        gradModel = Model(inputs = self.model.inputs, outputs = [self.model.get_layer(self.layer).output, self.model.output])
        with tf.GradientTape() as t:
            (featureMaps, predictions) = gradModel(image)
            if c == 'None':
                c = np.argmax(predictions[0])
            c = int(c)
            score = predictions[:,c]
        grads = t.gradient(score, featureMaps)
        first_y = np.exp(score)*grads
        second_y = np.exp(score)*grads*grads
        third_y = np.exp(score)*grads*grads*grads
        alpha_c_k_i_j_dem = second_y*2 + tf.reduce_sum(featureMaps)*third_y
        alpha_denom = np.where(alpha_c_k_i_j_dem != 0.0, alpha_c_k_i_j_dem, np.ones(alpha_c_k_i_j_dem.shape))
        alpha_c_k_i_j = second_y / alpha_denom
        w_c_k = tf.reduce_sum(alpha_c_k_i_j*np.maximum(0,grads.numpy()[0]), axis=(1,2))
        locMap = np.maximum(tf.reduce_sum(tf.multiply(w_c_k, featureMaps), axis=3).numpy()[0], 0)
        # 14 x 14 Map
        return locMap, c

def preProcessImage(imagePath):

    image = I.load_img(imagePath, target_size=(224,224))
    image = I.img_to_array(image)
    image = np.reshape(image,(1, 224,224,3))
    image = preprocess_input(image)
    return image

def preProcessImageArray(image):

    image = tf.image.resize(image, (224, 224))
    image = np.array(image)
    image = np.reshape(image,(1, 224,224,3))
    image = preprocess_input(image)
    return image

def parseArgs():
    parser = argparse.ArgumentParser(description='GradCAM')
    parser.add_argument('--imagePath', default='None', type=str)
    parser.add_argument('--imageClass', default='None', type=str)
    parser.add_argument('--folderPath', default='images/', type=str)
    parser.add_argument('--resultsPath', default='None', type=str)
    parser.add_argument('--layer', default='block5_conv3', type=str)
    parser.add_argument('--tolerance', default=30, type=int)
    parser.add_argument('--gradcampp', action='store_true')

    return parser.parse_args()


def mainMultipleImages(args):

    model = VGG16(weights='imagenet')

    gradCAM = 0
    if args.gradcampp:
        gradCAM = GradCAMPlusPlus(model, args.layer)
    else:
        gradCAM = GradCAM(model, args.layer)

    resultsFile = open('./evaluation.txt', 'a')

    t_drop = 0
    t_inc = 0
    n_im = 0

    pbar = tqdm(total=3923)
    for root, dirs, files in os.walk(args.folderPath):
            for name in files:
                path = os.path.join(root, name)
                image = preProcessImage(path)
                locMap, c = gradCAM.getLocalizationMap(image)
                heatMap = gradCAM.getHeatmap(locMap, image)

                drop, inc = gradCAM.evaluate(locMap, path, c, args.tolerance)
                t_drop += drop
                t_inc += inc
                n_im += 1

                if not args.resultsPath == 'None':
                    plt.imsave(args.resultsPath + args.layer + '_' + name, np.uint8(heatMap))
                
                pbar.update()

    resultsFile.write(args.layer + '\t' + str(t_drop/n_im) +'\t' + str(t_inc/n_im) + '\n')

    resultsFile.close()

def mainSimpleImage(args):

    model = VGG16(weights='imagenet')

    gradCAM = 0
    if args.gradcampp:
        gradCAM = GradCAMPlusPlus(model, args.layer)
        print('Gradcam++')
    else:
        gradCAM = GradCAM(model, args.layer)
        print('Gradcam')


    path = args.imagePath
    image = preProcessImage(path)
    c = args.imageClass
    locMap, c = gradCAM.getLocalizationMap(image, c)
    heatMap = gradCAM.getHeatmap(locMap, image)
    name = os.path.split(path)[-1]
    plt.imsave(args.resultsPath + args.layer + '_' + str(c) + '_' + name, np.uint8(heatMap))
    

def mainMultipleLayers(args):

    layers = ['block1_conv1', 'block1_conv2','block2_conv1','block2_conv2','block3_conv1','block3_conv2','block3_conv3','block4_conv1','block4_conv2','block4_conv3','block5_conv1','block5_conv2','block5_conv3']
    
    for layer in layers:
        print('Layer:',layer)
        args.layer = layer
        if not args.imagePath == 'None':
            mainSimpleImage(args)
        elif not args.folderPath == 'None':    
            mainMultipleImages(args)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    args = parseArgs()

    if args.layer == 'list':
        mainMultipleLayers(args)
    elif not args.imagePath == 'None':
        mainSimpleImage(args)
    elif not args.folderPath == 'None':
        mainMultipleImages(args)


        

if __name__ == "__main__":
    main()

