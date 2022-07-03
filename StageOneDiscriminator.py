from keras.layers import Concatenate
import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Embedding, MaxPooling2D, MaxPooling1D
from keras.layers import LeakyReLU, Activation
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Reshape
from tensorflow.keras.optimizers import Adam
from numpy.random import randint

import numpy as np
from numpy.random import default_rng
from numpy import ones

class StageOneDiscriminator(object):
    def __init__(self, img_shape=(64,64,3), text_shape=100):
        self.img_shape = img_shape
        self.text_shape = text_shape
        inputText = Input(shape=(text_shape,))
        text_dense = Dense(512)(inputText)
   

        inputImage = Input(shape=img_shape)
        x = Conv2D(64, kernel_size=(5,5), padding='same')(inputImage)
        x = Activation('tanh')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)

        x = Conv2D(128, kernel_size=5)(x)
        x = Activation('tanh')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)

        x = Conv2D(256, kernel_size=5)(x)
        x = Activation('tanh')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)

        #x = Conv2D(256, kernel_size=5)(x)
        #x = Activation('tanh')(x)
        #x = MaxPooling2D(pool_size=(2,2))(x)

        #x = Conv2D(256, kernel_size=5)(x)
        #x = Activation('tanh')(x)
        #x = MaxPooling2D(pool_size=(2,2))(x)
        
        
        x = Flatten()(x)
        x = Dense(1024)(x)

        combined = concatenate([x, text_dense])
        #changed for testing, x should be combined
        y = Activation('tanh')(combined)
        y = Dense(1)(y)
        y = Activation('sigmoid')(y)

        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        self.model = Model(inputs=[inputImage, inputText], outputs=y)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])