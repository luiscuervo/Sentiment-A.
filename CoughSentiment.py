import os
import sys
import numpy as np
import pandas as pd
from pydub import AudioSegment

from tensorflow.keras import models, layers
#from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import applications

import scipy.io.wavfile
import python_speech_features

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(1, DIR_PATH)
from db_utils import PlotMfcc, from_mp3_file_to_wav, DataFetcherSentimentSpeech


def create_wavs():
    json_file = DIR_PATH + '/dataset-main/metadata.json'
    df = pd.read_json(json_file)
    df = df.sample(frac=1).reset_index(drop=True)   #Arrange rows in random order

    for file in df.filename:    #Create a new directory with the sounds in .wav format
        if file == 'a9ecaf03-40a5-4b43-aaf3-f076f84a69aa.mp3':
            print('file unreadable')
        elif file == '098d66e5-bda6-4e99-b787-ab890046c44b.mp3':
            print('file unreadable')
        else:
            from_mp3_file_to_wav(file,DIR_PATH +'/dataset-main/raw',DIR_PATH +'/dataset-main/WAV')

    return None


def get_cough_files():
    '''Obtain Audio features '''

    json_file = DIR_PATH + '/dataset-main/metadata.json'
    df = pd.read_json(json_file)

    df = df[df.filename != '098d66e5-bda6-4e99-b787-ab890046c44b.mp3']      #we delete damaged data
    df = df[df.filename != 'a9ecaf03-40a5-4b43-aaf3-f076f84a69aa.mp3']

    df = df.sample(frac=1).reset_index(drop=True)                       #Arrange rows in random order

    wavs=[]
    for file in df.filename:    #change the filename pandas series to .wav name
        wavs.append(str(file.replace('.mp3', '.wav')))

    Y=[]
    for c in df.covid19:        #We consider that covid does not sound angry
        if c == True:
            Y.append([0])
        elif c==False:
            Y.append([1])

    print('m =', len(wavs))

    return wavs, Y


def get_cough_data():
    '''Returns the mfcc of the files (x) and its correspondig sentiment label (y)'''
    files, Y = get_cough_files()
    audio_data = DataFetcherSentimentSpeech(DIR_PATH, 50)


    X = []
    for file in files:
        file = DIR_PATH + '/dataset-main/WAV/' + file

        sound_file = AudioSegment.from_wav(file)
        audio = sound_file.get_array_of_samples()
        # audio[0] = sound_file.frame_rate
        audios = audio_data.get_mcc_from_audio(np.asarray(audio), sound_file.frame_rate, 500)
        # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
        X.append(audios[0])

    X = np.asarray(X)
    Y = np.asarray(Y)

    return X, Y

def import_cough_dataset():
    X, Y = get_cough_data()
    m = len(X)

    X_train, Y_train = X[0:int(m*0.7)], Y[0:int(m*0.7)]  # 70% training
    X_val, Y_val = X[int(m*0.7):int(m*0.85)], Y[int(m*0.7):int(m*0.85)]  #15% Validation
    X_test, Y_test = X[int(m*0.85):m], Y[int(m*0.85):m]       # 15% testing

    X_train, X_test, X_val = normalize_dataset(X_train, X_test, X_val)
    (X_train, Y_train), (X_test, Y_test), (X_val, Y_val) = reshape_dataset(X_train, Y_train, X_test, Y_test, X_val, Y_val)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def normalize_dataset(X_train, X_test, X_val):
    """Normalize speech recognition and computer vision datasets."""
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = (X_train-std)/mean
    X_test = (X_test-std)/mean
    X_val = (X_val-std)/mean

    return X_train, X_test, X_val

def reshape_dataset(X_train, Y_train, X_test, Y_test, X_val, Y_val):
    """Reshape dataset for Convolution."""
    num_pixels = X_test.shape[1]*X_test.shape[2]

    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1).astype('float32')

    #Ys are already in a to_categorical fashion

    #Y_train = to_categorical(Y_train)
    #Y_test = to_categorical(Y_test)

    return (X_train, Y_train), (X_test, Y_test), (X_val, Y_val)

def train_model(model, epochs, batch_size, X_train, y_train, X_val, Y_val, X_test, y_test):
    """Generic Deep Learning Model training function."""

    model.fit(X_train, y_train, validation_data=(X_val,Y_val), epochs=epochs,
              batch_size=batch_size, verbose=1)
    scores = model.evaluate(X_test, y_test, verbose=2)

    print("Baseline Error: %.2f%%" % (100-scores[1]*100))

    return model


def main():
    #create_wavs()      #Use this function to create wav files from the mp3 files

    # Hyperparameters

    epochs = 6
    batch_size = 220
    lr = 0.0001
    opt = Adam(lr=lr)

    # Import Datasets
    (X_train, Y_train, X_val, Y_val, X_test, Y_test) = import_cough_dataset()

    #Load model
    loaded_model = models.load_model(DIR_PATH+'/models/model_SA_1Out_a1.h5')

    print("Loaded model from disk")

    #To freeze layers all layers except the last (dense):
    for layer in range(len(loaded_model.layers)-1):
        loaded_model.layers[layer].trainable = False


    loaded_model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])


    trained_model = train_model(loaded_model, epochs, batch_size, X_train, Y_train, X_val, Y_val, X_test, Y_test)

    model_name = "SAtransfered_frozen"

    # Save model to h5 file
    trained_model.save('models/model_%s_a1.h5' %model_name)
    #model.save_weights('SAweights.h5')

    print("Saved model to disk")




    return None

if __name__ == '__main__':
    main()