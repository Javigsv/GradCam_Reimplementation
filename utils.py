
import tensorflow.keras.preprocessing.image as I

def preProcessImage(imagePath):
    image = I.load_img(imagePath, target_size=(224,224))
    image = I.img_to_array(image)
    image = np.reshape(image,(1, 224, 224, 3))
    image = preprocess_input(image)
    return image