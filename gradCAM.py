from tensorflow.keras.models import Model
import tensorflow as tf 
import numpy as np 
import cv2
import tensorflow.keras.preprocessing.image as I


class GradCAM:
    def __init__(self, model, layer, img_height, img_width):
        self.model = model
        self.layer = layer
        self.img_height = img_height
        self.img_width = img_width

    def getLocalizationMap(self, image, c = None):
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
        locMapResized = cv2.resize(locMap, (self.img_width, self.img_height))
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
        image = I.load_img(imagePath,target_size=(self.img_height,self.img_width))
        image = I.img_to_array(image)
        locMapResized = cv2.resize(locMap, (self.img_height,self.img_width))
        heatmap = locMapResized / np.max(locMapResized)
        _, thresh_img = cv2.threshold(np.uint8(heatmap*255), 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        black_img = np.zeros(image.shape).astype(image.dtype)
        color = [1, 1, 1]
        cv2.fillPoly(black_img, contours, color)
        segmented_img = image*black_img
        segmented_img = np.reshape(segmented_img,(1, self.img_height, self.img_width, 3))
        segmented_img = preprocess_input(segmented_img)
        original_img = np.reshape(image,(1, self.img_height, self.img_width, 3))
        original_img = preprocess_input(original_img)
        Y = self.model(original_img)[0,c].numpy()
        O = self.model(segmented_img)[0,c].numpy()
        drop = np.maximum(0,Y-O)/Y
        inc = 0
        if Y < O:
            inc = 1
        return drop, inc