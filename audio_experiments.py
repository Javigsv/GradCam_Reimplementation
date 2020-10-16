import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import os
import pickle
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input 
from tensorflow.keras.applications.resnet import ResNet50
import tensorflow.keras.preprocessing.image as I

from time import gmtime, asctime

from GradCAM import GradCAM
from GradCAMPlusPlus import GradCAMPlusPlus



<<<<<<< HEAD
=======

>>>>>>> main

def parseArgs():
    parser = argparse.ArgumentParser(description='GradCAM')
    parser.add_argument('--imagePath', default='None', type=str)
    parser.add_argument('--modelPath', default='model/model_songsWithTags_20000.h5', type=str)
    parser.add_argument('--imageClass', default='None', type=str)
    parser.add_argument('--folderPath', default='images/', type=str)
    parser.add_argument('--resultsPath', default='None', type=str)
    parser.add_argument('--layer', default='last', type=str)
    return parser.parse_args()

"""
def mainMultipleImages(args):

    resultsFile = open('./eval/evaluation'+ asctime(gmtime()).replace(' ','_').replace(':','')+'.txt', 'a')
    model = tf.keras.models.load_model(args.modelPath)
    is_conv = 'conv'
    
    #resultsFile.write(args.model + '\n')

    if args.layer == 'all':
        for layer in model.layers:
            if is_conv in layer.name:
                print('Evaluating on layer: ' + layer.name)
                gradCAM = GradCAM(model, layer.name)
                t_drop = 0
                t_inc = 0
                n_im = 0
                for root, _, files in os.walk(args.folderPath):
                    for name in files:
                        path = os.path.join(root, name)
                        image = preProcessImage(path)
                        locMap, c, _ = gradCAM.getLocalizationMap(image)
                        drop, inc = gradCAM.evaluate(locMap, path, c)
                        t_drop += drop
                        t_inc += inc
                        n_im += 1
                        if args.resultsPath != 'None':
                            heatMap = gradCAM.getHeatmap(locMap, image)
                            plt.imsave(args.resultsPath + layer.name + '_' + name, np.uint8(heatMap))
                resultsFile.write(layer.name + '\t' + str(t_drop/n_im) +'\t' + str(t_inc/n_im) + '\n')
        resultsFile.close()
    else:
        if args.layer == "last":
            conv_layers = []
            for layer in model.layers:
                if is_conv in layer.name:
                    conv_layers.append(layer.name)
            # Take last layer 
            layer_name = conv_layers[-1]
        else:
            layer_name = args.layer

        gradCAM = GradCAM(model, layer_name)
        t_drop = 0
        t_inc = 0
        n_im = 0
        for root, _, files in os.walk(args.folderPath):
            for name in files:
                path = os.path.join(root, name)
                image = preProcessImage(path)
                locMap, c, _ = gradCAM.getLocalizationMap(image)
                drop, inc = gradCAM.evaluate(locMap, path, c)
                t_drop += drop
                t_inc += inc
                n_im += 1
                if args.resultsPath != 'None':
                    heatMap = gradCAM.getHeatmap(locMap, image)
                    plt.imsave(args.resultsPath + layer_name + '_' + name, np.uint8(heatMap))
        resultsFile.write(layer_name + '\t' + str(t_drop/n_im) +'\t' + str(t_inc/n_im) + '\n')
        resultsFile.close()
"""

def mainSimpleImage(args, input_height, input_width):
    model = tf.keras.models.load_model(args.modelPath)
    is_conv = 'conv'
        
    classMap = pickle.load(open('classMap.p', 'rb'))
    
    c = None
    if args.imageClass != 'None':
        c = classMap[args.imageClass]

    if args.layer == 'all':
        for layer in model.layers:
            if is_conv in layer.name:
                gradCAM = GradCAM(model, layer.name, input_height, input_width)
                path = args.imagePath
                image = preProcessImage(path, )
                c = classMap.get(args.imageClass, args.imageClass)
                locMap, _, _ = gradCAM.getLocalizationMap(image, c)
                if args.resultsPath != 'None':
                    heatMap = gradCAM.getHeatmap(locMap, image)
                    name = os.path.split(path)[-1]
                    plt.imsave(args.resultsPath + layer.name + '_' + str(c) + '_' + name, np.uint8(heatMap))
    else:
        if args.layer == "last":
            conv_layers = []
            for layer in model.layers:
                if is_conv in layer.name:
                    conv_layers.append(layer.name)
            # Take last layer 
            layer_name = conv_layers[-1]
        else:
            layer_name = args.layer

        gradCAM = GradCAM(model, layer_name, input_height, input_width)
        path = args.imagePath
        image = preProcessImage(path)   
        locMap, _, _ = gradCAM.getLocalizationMap(image, c)
        #gradCAMpp = GradCAMPlusPlus(model, layer_name, input_height, input_width)
        #locMappp, _= gradCAMpp.getLocalizationMap(image, c)

        if args.resultsPath != 'None':
            heatMap = gradCAM.getHeatmap(locMap, image)
            name = os.path.split(path)[-1]
            plt.imsave(args.resultsPath + layer_name + '_' + str(c) + name, np.uint8(heatMap))
            #plt.imsave(args.resultsPath + layer_name + '_++' + str(c) + name, np.uint8(gradCAMpp.getHeatmap(locMappp, image)))


def main():
    input_width = 431
    input_height = 228
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    args = parseArgs()
    if args.imagePath != 'None':
        mainSimpleImage(args, input_height, input_width)
    #elif args.folderPath != 'None':
    #    mainMultipleImages(args)
        

if __name__ == "__main__":
    main()
