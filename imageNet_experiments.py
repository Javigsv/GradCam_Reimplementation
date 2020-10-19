import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import os
import pickle
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input 
from tensorflow.keras.applications.resnet import ResNet50
import tensorflow.keras.applications.resnet as resnet
import tensorflow.keras.preprocessing.image as I

from time import gmtime, asctime

from GradCAM import GradCAM

from GradCAMPlusPlus import GradCAMPlusPlus

def preProcessImage(imagePath, model, input_height, input_width):
    """
    Reshape the image to input_height and input_widht as it is a models' constraint
    """
    image = I.load_img(imagePath, target_size=(input_height, input_width))
    image = I.img_to_array(image)
    image = np.reshape(image,(1, input_height, input_width,3))
    if model == 'VGG16':
        image = preprocess_input(image)
    elif model == 'ResNet50':
        image = resnet.preprocess_input(image)
    return image


def parseArgs():
    parser = argparse.ArgumentParser(description='GradCAM')
    parser.add_argument('--imagePath', default='None', type=str)
    parser.add_argument('--model', default='VGG16', type=str)
    parser.add_argument('--imageClass', default='None', type=str)
    parser.add_argument('--folderPath', default='images/', type=str)
    parser.add_argument('--resultsPath', default='None', type=str)
    parser.add_argument('--layer', default='last', type=str)
    parser.add_argument('--gradCAM', default=True, type=bool)
    parser.add_argument('--gradCAMpp', default=False, type=bool)
    return parser.parse_args()


def mainMultipleImages(args, input_height, input_width):
    """
    """
    resultsFile = open('./eval/evaluation'+ asctime(gmtime()).replace(' ','_').replace(':','')+'.txt', 'a')
    if args.model == 'VGG16':
        model = VGG16(weights='imagenet')
        is_conv = 'conv'
        resultsFile.write(args.model + '\n')
    if args.model == 'ResNet50':
        model = ResNet50(weights='imagenet')
        is_conv = '_conv'
        resultsFile.write(args.model + '\n')

    if args.layer == 'all':
        for layer in model.layers:
            if is_conv in layer.name:
                print('Evaluating on layer: ' + layer.name)
                gradCAM = GradCAM(model, layer.name, input_height, input_width)
                t_drop = 0
                t_inc = 0
                n_im = 0
                for root, _, files in os.walk(args.folderPath):
                    for name in files:
                        path = os.path.join(root, name)
                        image = preProcessImage(path, args.model, input_height, input_width)
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

        gradCAM = GradCAM(model, layer_name, input_height, input_width)
        t_drop = 0
        t_inc = 0
        n_im = 0
        for root, _, files in os.walk(args.folderPath):
            for name in files:
                path = os.path.join(root, name)
                image = preProcessImage(path, args.model, input_height, input_width)
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


def mainSimpleImage(args, input_height, input_width):
    """
    Perform gradCAM over a single image specified in args.imagePath. The result will be saved if
    args.resultsPath has been specified.
    """
    if args.model == 'VGG16':
        model = VGG16(weights='imagenet')
        is_conv = 'conv'
    if args.model == 'ResNet50':
        model = ResNet50(weights='imagenet')
        is_conv = '_conv'
    
    classMap = pickle.load(open('classMap.p', 'rb'))
    
    c = None

    if args.imageClass != 'None':
        c = classMap[args.imageClass]

    if args.layer == 'all':
        for layer in model.layers:
            if is_conv in layer.name:
                path = args.imagePath
                image = preProcessImage(path, args.model, input_height, input_width)
                c = classMap.get(args.imageClass, args.imageClass)
                
                if args.gradCAM:
                    gradCAM = GradCAM(model, layer.name, input_height, input_width)
                    locMap, _, _ = gradCAM.getLocalizationMap(image, c)
                
                if args.gradCAMpp:
                    gradCAM_pp = GradCAMPlusPlus(model, layer.name, input_height, input_width)
                    locMap_pp, _ = gradCAM_pp.getLocalizationMap(image, c)
                
                if args.resultsPath != 'None':
                    name = os.path.split(path)[-1]
                    if args.gradCAM:
                        heatMap = gradCAM.getHeatmap(locMap, image)
                        plt.imsave(args.resultsPath + layer.name + '_' + str(c) + '_' + name, np.uint8(heatMap))
                    
                    if args.gradCAMpp:
                        heatMap_pp = gradCAM_pp.getHeatmap(locMap_pp, image)
                        plt.imsave(args.resultsPath + layer.name + '_++_' + str(c) + name, np.uint8(heatMap_pp))
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
        
        path = args.imagePath
        image = preProcessImage(path, args.model, input_height, input_width)   
        
        if args.gradCAM:
            gradCAM = GradCAM(model, layer_name, input_height, input_width)    
            locMap, _, _ = gradCAM.getLocalizationMap(image, c)
            
        if args.gradCAMpp:
            gradCAM_pp = GradCAMPlusPlus(model, layer_name, input_height, input_width)
            locMap_pp, _ = gradCAM_pp.getLocalizationMap(image, c)

        if args.resultsPath != 'None':
            name = os.path.split(path)[-1] 
            if args.gradCAM:
                heatMap = gradCAM.getHeatmap(locMap, image)
                plt.imsave(args.resultsPath + layer_name + '_' + str(c) + name, np.uint8(heatMap))
            if args.gradCAMpp:
                heatMap_pp = gradCAM_pp.getHeatmap(locMap_pp, image)
                plt.imsave(args.resultsPath + layer_name + '_++_' + str(c) + name, np.uint8(heatMap_pp))


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    args = parseArgs()
    input_height = 224
    input_width = 224
    if args.imagePath != 'None':
        mainSimpleImage(args, input_height, input_width)
    elif args.folderPath != 'None':
        mainMultipleImages(args, input_height, input_width)
        

if __name__ == "__main__":
    main()

