import os
import sys
import numpy as np
from tensorflow.keras import models, layers
#from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import applications
import pandas as pd

#from matplotlib import pyplot as plt

# Change this to the location of the database directories
DB_DIR = os.path.dirname(os.path.realpath(__file__))

# Import databases
sys.path.insert(1, DB_DIR)
from db_utils import get_sentiment_data, PlotMfcc

def Secure_Voice_Channel(func):
    """Define Secure_Voice_Channel decorator."""
    def execute_func(*args, **kwargs):
        print('Established Secure Connection.')
        returned_value = func(*args, **kwargs)
        print("Ended Secure Connection.")

        return returned_value

    return execute_func

@Secure_Voice_Channel

def generic_vns_function(input_shape, units, lr, classes=8):
    """Generic Deep Learning Model generator."""

    #base_model = applications.ResNet50V2(include_top=False, weights=None, input_shape=input_shape, pooling=None,
    #                                     classes=classes)

    base_model = applications.ResNet50V2(include_top=False, weights=None, input_shape=input_shape, pooling=None)

    x = base_model.output

    #x = layers.GlobalAveragePooling2D()(x)     #This is used in MainCovid.py

    x = layers.Flatten()(x)                     #I am using Flatten because I am more familiar with it

    #x = layers.Dense(units, activation = 'relu')(x)    #MainCovid.py has two final dense layers, but we will only have one

    predictions = layers.Dense(units, activation= 'sigmoid')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)

    opt = Adam(lr=lr)

    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    return model

def train_model(model, epochs, batch_size, X_train, y_train, X_test, y_test):
    """Generic Deep Learning Model training function."""

    model.fit(X_train, y_train, validation_data=None, epochs=epochs,
              batch_size=batch_size, verbose=1)                         
    scores = model.evaluate(X_test, y_test, verbose=2)

    print("Baseline Error: %.2f%%" % (100-scores[1]*100))

    return model

def import_sentiment_dataset():
    X, Y = get_sentiment_data()
    m = len(X)

    X_train, Y_train = X[0:int(m*0.8)], Y[0:int(m*0.8)]     # 80% training
    X_test, Y_test = X[int(m*0.8):m], Y[int(m*0.8):m]       # 20% testing

    #Maybe I should implement validation data, but I still don't know how that works honestly

    X_train, X_test = normalize_dataset(X_train, X_test)
    (X_train, Y_train), (X_test, Y_test) = reshape_dataset(X_train, Y_train, X_test, Y_test)
    return X_train, Y_train, X_test, Y_test

def normalize_dataset(X_train, X_test):
    """Normalize speech recognition and computer vision datasets."""
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = (X_train-std)/mean
    X_test = (X_test-std)/mean
    return X_train, X_test

def reshape_dataset(X_train, Y_train, X_test, Y_test):
    """Reshape dataset for Convolution."""
    num_pixels = X_test.shape[1]*X_test.shape[2]

    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')

    #Ys are already in a to_categorical fashion

    #Y_train = to_categorical(Y_train)
    #Y_test = to_categorical(Y_test)

    return (X_train, Y_train), (X_test, Y_test)


def main():

    # Hyperparameters

    layer_units = 1000
    epochs = 5
    batch_size = 200
    lr = 0.0001

    # Import Datasets
    (X_train, Y_train, X_test, Y_test) = import_sentiment_dataset()

    #print(X_test.shape[1:])
    #print(Y_train[1])
    #PlotMfcc(X_test[0])


    # Generate and train model
    model = generic_vns_function(X_train.shape[1:], Y_train.shape[1], layer_units, lr)
    #print(model.summary())

    trained_model = train_model(model, epochs, batch_size, X_train, Y_train, X_test, Y_test)

    model_name = "1st attempt"

    # Save model to h5 file
    trained_model.save('models/model_%s_a1.h5' %model_name)

    return None

if __name__ == '__main__':
    main()

