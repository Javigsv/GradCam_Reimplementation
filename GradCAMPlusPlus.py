from gradCAM import GradCAM
from tensorflow.keras.models import Model
import tensorflow as tf 
import numpy as np 


class GradCAMPlusPlus(GradCAM):
    def __init__(self, model, layer):
        super().__init__(model, layer)

    def getLocalizationMap(self, image, c = 'None'):
        gradModel = Model(inputs = self.model.inputs, outputs = [self.model.get_layer(self.layer).output, self.model.output])
        with tf.GradientTape() as t:
            (featureMaps, predictions) = gradModel(image)
            if c == None:
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
