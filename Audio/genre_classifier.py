import tensorflow as tf
import keras
from keras import layers
from keras.layers import Activation, Dense, Dropout, Conv1D, Conv2D, Flatten, BatchNormalization, ZeroPadding2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import SGD
import keras.backend as K
from keras.models import load_model
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
import numpy as np
from numpy import argmax
import pandas as pd
import random

import warnings
warnings.filterwarnings('ignore')


def genre_number(i):
    """
    Assign a number to each genre type
    """
    if i == 'Hip-Hop':
        return 0
    elif i == 'Pop':
        return 1
    elif i == 'Folk':
        return 2
    elif i == 'Rock':
        return 3
    elif i == 'Experimental':
        return 4
    elif i == 'International':
        return 5
    elif i == 'Electronic':
        return 6
    else:
        return '7' #instrumental


def load_reference_table(file_name = 'music_analysis.csv', root_folder = "songsWithTags_wav"):
    """
    Load table that contain the genres and the audio files paths.  
    """
    ref_table = pd.read_csv(file_name)
    ref_table['file_name'] = ref_table['file_name'].apply(lambda x: '{0:0>6}'.format(x))
    ref_table['genre_number'] = ref_table['genre'].apply(genre_number)
    ref_table['path'] = root_folder + "/" + ref_table['file_name'].astype('str') + ".wav"
    return ref_table 


def get_augment_time_strech_spect(y, sr, rate):
    """
    Augment audio signal by applying time streching and get its melspectrogram.
    """
    y_augmented = librosa.effects.time_stretch(y, rate=rate)
    y_augmented_spect = librosa.feature.melspectrogram(y=y_augmented, sr=sr, n_mels=128)
    return y_augmented_spect


def get_augment_pitch_shift_spect(y, sr, n_steps):
    """
    Augment audio signal by applying pitch shifting and get its melspectrogram
    """
    y_augmented = librosa.effects.pitch_shift(y, sr, n_steps=n_steps)
    y_augmented_spect = librosa.feature.melspectrogram(y=y_augmented, sr=sr, n_mels=128)
    return y_augmented_spect


def generate_dataset(ref_table, data_augmentation=False):
    """
    The dataset consists on the spectrograms of the original signal and also of four different
    augmentations (2 time strechi g and 2 pitch shifting)
    """
    dataset = []
    for song in ref_table.itertuples():
        audio_signal, sr = librosa.load(song.path)  

        spectrogram = librosa.feature.melspectrogram(audio_signal, sr, n_mels=128)
        dataset.append((spectrogram, song.genre_number))
        
        if data_augmentation:
            spect_augment_1 = get_augment_time_strech_spect(audio_signal, sr, rate = 0.8)
            dataset.append((spect_augment_1, song.genre_number))

            spect_augment_2 = get_augment_time_strech_spect(audio_signal, sr, rate = 0.9)
            dataset.append((spect_augment_2, song.genre_number))

            spect_augment_3 = get_augment_pitch_shift_spect(audio_signal, sr, n_steps = 2)
            dataset.append((spect_augment_3, song.genre_number))
            
            spect_augment_4 = get_augment_pitch_shift_spect(audio_signal, sr, n_steps = -2)
            dataset.append((spect_augment_4, song.genre_number))
        
    return dataset


def split_dataset(dataset):
    """
    Dataset will be splitted in train (80%), validation(10%) and test (10%) and reformat for being a valid 
    input for the CNN (reshaping of input and one hot encoding of classes)
    """
    random.shuffle(dataset)

    #train val test split 8:1:1
    N_samples = len(dataset)
    N_samples_train = N_samples*0.8
    N_samples_val = N_samples*0.1
    
    train = dataset[:round(N_samples_train)]
    val = dataset[N_samples_train:round(N_samples_train + N_samples_val)]
    test = dataset[round(N_samples_val + N_samples_train):]

    X_train, Y_train = zip(*train)
    X_val, Y_val = zip(*val)
    X_test, Y_test = zip(*test)

    # Reshape for CNN input
    X_train = np.array([x.reshape((128, 431, 1)) for x in X_train])
    X_val = np.array([x.reshape((128, 431, 1)) for x in X_val])
    X_test = np.array([x.reshape((128, 431, 1)) for x in X_test])

    # One-Hot encoding for classes
    Y_train = np.array(keras.utils.to_categorical(Y_train, 8))
    Y_val = np.array(keras.utils.to_categorical(Y_val, 8))
    Y_test = np.array(keras.utils.to_categorical(Y_test, 8))

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def create_CNN():
    """
    CNN with 3 convolution + pooling layers
    """
    model = Sequential()
    input_shape=(128, 431, 1)

    model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
    model.add(AveragePooling2D((2, 2), strides=(2,2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="same"))
    model.add(AveragePooling2D((2, 2), strides=(2,2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="same"))
    model.add(AveragePooling2D((2, 2), strides=(2,2)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(rate=0.5))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))

    model.add(Dense(8))
    model.add(Activation('softmax'))

    return model


def plot_acc_loss(hist, file_name="acc_loss.png"):
    """Save figure that shows training and validations accuracies and losses"""
    plt.figure(figsize=(12,8))
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.legend(['loss','val_loss', 'acc','val_acc'])
    plt.savefig(file_name)

    

def main(saved_model_file = 'music_genre_classification.h5'):
    ref_table = load_reference_table()
    dataset = generate_dataset(ref_table)
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_dataset(dataset)
    model = create_CNN()

    epochs = 100
    batch_size = 64
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)
    hist = model.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=batch_size, validation_data= (X_val, Y_val), callbacks=[early_stopping, tb_hist]) 

    plot_acc_loss(hist)
    
    score = model.evaluate(x=X_test, y=Y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save(saved_model_file)


if __name__ == "__main__":
    main()
