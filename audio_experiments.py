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
from tqdm import tqdm

from genreClassifier.model import genre_number

from GradCAM import GradCAM
from GradCAMPlusPlus import GradCAMPlusPlus


def parseArgs():
    parser = argparse.ArgumentParser(description='audioExperiments')
    parser.add_argument('--oneSpectrogram', default=True, type=bool)
    parser.add_argument('--modelPath', default='./genreClassifier/output/model_songsWithTags_2000.h5', type=str)
    parser.add_argument('--resultsPath', default='None', type=str)
    parser.add_argument('--layer', default='last', type=str)
    parser.add_argument('--genre', default=None, type=str)
    parser.add_argument('--gradCAM', default=True, type=bool)
    parser.add_argument('--gradCAMpp', default=False, type=bool)
    return parser.parse_args()

def mainAllSpectrograms(args, dataset, is_conv = 'conv'):
    model = tf.keras.models.load_model(args.modelPath)
    
    # We get the GradCAM for all spectrograms in dataset 
    for sample_idx in tqdm(range(len(dataset))):
        image, genre_idx = dataset[sample_idx]
        input_height, input_width = image.shape
        image = np.reshape(image, (1, input_height, input_width, 1))
    
        perform_GradCAM(args, model, image, is_conv, int(genre_idx), sample_idx)


def mainOneSpectrogram(args, dataset, is_conv = 'conv'):

    model = tf.keras.models.load_model(args.modelPath)
    
    input_height, input_width = dataset[0][0].shape
    
    random.shuffle(dataset)  

    image = np.array([])
    sample_idx = 0
    args_genre_number = genre_number(args.genre)
    while len(image) == 0:
        if dataset[sample_idx][1] == args_genre_number:
            image = dataset[sample_idx][0]
    
    image = np.reshape(image, (1, input_height, input_width, 1))
    
    perform_GradCAM(args, model, image, is_conv, args_genre_number)


    
def perform_GradCAM(args, model, image, is_conv, genre_idx, id=0):
    _, input_height, input_width, _ = image.shape
    if args.layer == 'all':
        for layer in model.layers:
            if is_conv in layer.name:  
                if args.gradCAM:
                    gradCAM = GradCAM(model, layer.name, input_height, input_width)
                    locMap, _, _ = gradCAM.getLocalizationMap(image)
                if args.gradCAMpp:
                    gradCAM_pp = GradCAMPlusPlus(model, layer.name, input_height, input_width)
                    locMap_pp, _ = gradCAM_pp.getLocalizationMap(image)
                
                if args.resultsPath != 'None':
                    if args.gradCAM:
                        heatMap = gradCAM.getHeatmap(locMap, image)
                        plt.imsave(args.resultsPath + layer.name + '_' + str(genre_idx) + "_" + str(id) + ".jpg", np.uint8(heatMap))
                    if args.gradCAMpp:
                        heatMap_pp = gradCAM_pp.getHeatmap(locMap_pp, image)
                        plt.imsave(args.resultsPath + layer.name + '_++_' + str(genre_idx) + "_" + str(id) + ".jpg", np.uint8(heatMap_pp))

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
                plt.imsave(args.resultsPath + layer_name + '_' + str(genre_idx) + "_" + str(id) + ".jpg", np.uint8(heatMap))
            if args.gradCAMpp:
                heatMap_pp = gradCAM_pp.getHeatmap(locMap_pp, image)
                plt.imsave(args.resultsPath + layer_name + '_++_' + str(genre_idx) + "_" + str(id) + ".jpg", np.uint8(heatMap_pp))


def main(input_spectrograms_file = 'genreClassifier/input_spectrograms_2000.pickle'):

    dataset = pickle.load(open(input_spectrograms_file, 'rb'))

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    args = parseArgs()

    if args.oneSpectrogram:
        mainOneSpectrogram(args, dataset)
    else:
        mainAllSpectrograms(args, dataset)

        

if __name__ == "__main__":
    main()
