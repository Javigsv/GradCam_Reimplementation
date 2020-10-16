import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import os
import pickle
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.models import Model
import tensorflow.keras.preprocessing.image as I
from time import gmtime, asctime

class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer

    def getLocalizationMap(self, image, c = 'None'):
        gradModel = Model(inputs = self.model.inputs, outputs = [self.model.get_layer(self.layer).output, self.model.output])
        with tf.GradientTape() as t:
            (featureMaps, predictions) = gradModel(image)
            if c == None:
                c = np.argmax(predictions[0])
            c = int(c)
            score = predictions[:,c]
        
        top_c = np.flip(np.array(predictions).argsort()[0][-10:])

        grads = t.gradient(score, featureMaps)
        alpha_c_k = tf.reduce_mean(grads, axis=(1,2))
        locMap = np.maximum(tf.reduce_sum(tf.multiply(alpha_c_k, featureMaps), axis=3).numpy()[0], 0)
        # 14 x 14 Map
        return locMap, c, top_c

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
    
    def evaluate(self, locMap, imagePath, c):
        image = I.load_img(imagePath,target_size=(224,224))
        image = I.img_to_array(image)
        locMapResized = cv2.resize(locMap, (224, 224))
        heatmap = locMapResized / np.max(locMapResized)
        _, thresh_img = cv2.threshold(np.uint8(heatmap*255), 30, 255, cv2.THRESH_BINARY)
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

def preProcessImage(imagePath):
    image = I.load_img(imagePath, target_size=(224,224))
    image = I.img_to_array(image)
    image = np.reshape(image,(1, 224,224,3))
    image = preprocess_input(image)
    return image


def parseArgs():
    parser = argparse.ArgumentParser(description='GradCAM')
    parser.add_argument('--imagePath', default=None, type=str)
    parser.add_argument('--model', default='binaryClassifier', type=str)
    parser.add_argument('--imageClass', default=None, type=str)
    parser.add_argument('--folderPath', default='images/', type=str)
    parser.add_argument('--resultsPath', default=None, type=str)
    parser.add_argument('--layer', default='last', type=str)
    return parser.parse_args()


def mainMultipleImages(args):

    resultsFile = open('./eval/evaluation'+ asctime(gmtime()).replace(' ','_').replace(':','')+'.txt', 'a')
    if args.model == 'binaryClassifier':
        model = tf.keras.models.load_model('audio/model/model_songsWithTags_20000.h5')
        is_conv = 'conv'

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
                        if not args.resultsPath == 'None':
                            heatMap = gradCAM.getHeatmap(locMap, image)
                            plt.imsave(args.resultsPath + layer.name + '_' + name, np.uint8(heatMap))
                resultsFile.write(layer.name + '\t' + str(t_drop/n_im) +'\t' + str(t_inc/n_im) + '\n')
        resultsFile.close()
    else:
        gradCAM = GradCAM(model, args.layer)
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
                if not args.resultsPath == 'None':
                    heatMap = gradCAM.getHeatmap(locMap, image)
                    plt.imsave(args.resultsPath + args.layer + '_' + name, np.uint8(heatMap))
        resultsFile.write(args.layer + '\t' + str(t_drop/n_im) +'\t' + str(t_inc/n_im) + '\n')
        resultsFile.close()

def mainSimpleImage(args):
    if args.model == 'binaryClassifier':
        model = tf.keras.models.load_model('audio/model/model_songsWithTags_20000.h5')
        is_conv = 'conv'

    gradCAM = GradCAM(model, args.layer)
    classMap = pickle.load(open('classMap.p', 'rb'))
    
    c = None
    if args.imageClass != None:
        c = classMap[args.imageClass]

    if args.layer == 'all':
        for layer in model.layers:
            if is_conv in layer.name:
                gradCAM = GradCAM(model, layer.name)
                path = args.imagePath
                image = preProcessImage(path)
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

        gradCAM = GradCAM(model, layer_name)
        path = args.imagePath
        image = preProcessImage(path)   
        locMap, _, _ = gradCAM.getLocalizationMap(image, c)
        if args.resultsPath != 'None':
            heatMap = gradCAM.getHeatmap(locMap, image)
            name = os.path.split(path)[-1]
            plt.imsave(args.resultsPath + args.layer + '_' + str(c) + name, np.uint8(heatMap))

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    args = parseArgs()
    if not args.imagePath == 'None':
        mainSimpleImage(args)
    elif not args.folderPath == 'None':
        mainMultipleImages(args)
        

if __name__ == "__main__":
    main()



