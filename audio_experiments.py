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
import random
from time import gmtime, asctime

from GradCAM import GradCAM
from GradCAMPlusPlus import GradCAMPlusPlus



def parseArgs():
    parser = argparse.ArgumentParser(description='audioExperiments')
    parser.add_argument('--oneSpectrogram', default=True, type=bool)
    parser.add_argument('--modelPath', default='model/model_songsWithTags_20000.h5', type=str)
    parser.add_argument('--resultsPath', default='None', type=str)
    parser.add_argument('--layer', default='last', type=str)
    parser.add_argument('--genre', default='Hip-Hop', type=str)
    parser.add_argument('--gradCAM', default=True, type=bool)
    parser.add_argument('--gradCAMpp', default=False, type=bool)
    return parser.parse_args()

"""
def mainAllSpectrograms(args, dataset, is_conv = 'conv'):

    resultsFile = open('./eval/evaluation'+ asctime(gmtime()).replace(' ','_').replace(':','')+'.txt', 'a')
    model = tf.keras.models.load_model(args.modelPath)
    
    
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

def mainOneSpectrogram(args, dataset, is_conv = 'conv'):

    model = tf.keras.models.load_model(args.modelPath)

    input_height, input_width = dataset[0][0].shape()
    
    random.shuffle(dataset)  

    image = None
    sample_idx = 0
    while image == None:
        if dataset[sample_idx][1] == args.genre:
            image = dataset[sample_idx][0]
    
    if args.layer == 'all':
        for layer in model.layers:
            if is_conv in layer.name:
                if args.gradCAM:
                    gradCAM = GradCAM(model, layer.name, input_height, input_width)
                    locMap, _, _ = gradCAM.getLocalizationMap(image)
                if args.gradCAMpp:
                    gradCAM_pp = GradCAMPlusPlus(model, layer.name, input_height, input_width)
                    locMap_pp, _= gradCAM_pp.getLocalizationMap(image)
                
                if args.resultsPath != 'None':
                    if args.gradCAM:
                        heatMap = gradCAM.getHeatmap(locMap, image)
                        plt.imsave(args.resultsPath + layer.name + '_' + args.genre, np.uint8(heatMap))
                    if args.gradCAMpp:
                        heatMap_pp = gradCAM_pp.getHeatmap(locMap_pp, image)
                        plt.imsave(args.resultsPath + layer.name + '_++_' + args.genre, np.uint8(heatMap_pp))

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

        if args.gradCAM:
            gradCAM = GradCAM(model, layer_name, input_height, input_width)
            locMap, _, _ = gradCAM.getLocalizationMap(image)
        if args.gradCAMpp:
            gradCAM_pp = GradCAMPlusPlus(model, layer_name, input_height, input_width)
            locMap_pp, _= gradCAM_pp.getLocalizationMap(image)

        if args.resultsPath != 'None':
            if args.gradCAM:
                heatMap = gradCAM.getHeatmap(locMap, image)
                plt.imsave(args.resultsPath + layer_name + '_' + args.genre, np.uint8(heatMap))
            if args.gradCAMpp:
                heatMap_pp = gradCAM_pp.getHeatmap(locMap_pp, image)
                plt.imsave(args.resultsPath + layer_name + '_++_' + args.genre, np.uint8(heatMap_pp))


def main(input_spectrograms_file = 'audio/input_spectrograms.pickle'):

    dataset = pickle.load(open(input_spectrograms_file, 'rb'))

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    args = parseArgs()

    if args.oneSpectrogram:
        mainOneSpectrogram(args, dataset)
    else:
        # mainAllSpectrograms(args, dataset)

        

if __name__ == "__main__":
    main()
