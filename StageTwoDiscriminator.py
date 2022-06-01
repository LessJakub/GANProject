import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate, Activation, Conv2D, Dense, Flatten,MaxPooling2D
from tensorflow.keras.optimizers import Adam
from numpy.random import randint
import numpy as np
from numpy import ones

class StageTwoDiscriminator(object):
    def __init__(self, img_shape=(256,256,3), text_shape=100):
        self.img_shape = img_shape
        self.text_shape = text_shape

        inputText = Input(shape=(text_shape,))
        text_dense = Dense(1024)(inputText)

        self.glove
   

        inputImage = Input(shape=img_shape)
        #256
        x = Conv2D(64, kernel_size=(5,5), padding='same')(inputImage)
        x = Activation('tanh')(x)
        #128
        x = MaxPooling2D(pool_size=(2,2))(x)
        #124
        x = Conv2D(128, kernel_size=5)(x)
        x = Activation('tanh')(x)
        #62
        x = MaxPooling2D(pool_size=(2,2))(x)
        #58
        x = Conv2D(256, kernel_size=5)(x)
        x = Activation('tanh')(x)
        #29
        x = MaxPooling2D(pool_size=(2,2))(x)
        #25
        x = Conv2D(128, kernel_size=5)(x)
        x = Activation('tanh')(x)
        #12
        x = MaxPooling2D(pool_size=(2,2))(x)
        #8
        x = Conv2D(64, kernel_size=5)(x)
        x = Activation('tanh')(x)
        #4
        x = MaxPooling2D(pool_size=(2,2))(x)
        
        
        x = Flatten()(x)
        x = Dense(1024)(x)

        combined = concatenate([x, text_dense])
        y = Activation('tanh')(combined)
        y = Dense(1)(y)
        y = Activation('sigmoid')(y)

        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        self.model = Model(inputs=[inputImage, inputText], outputs=y, name='DiscriminatorTwo')
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    def load_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=self.img_shape[2])
        img = tf.keras.layers.Resizing(self.img_shape[0], self.img_shape[1])(img)
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path

    def generate_real_samples(self, imageDataset, embedingsDataset, n_samples):
        ix = randint(0, imageDataset.shape[0], n_samples)
        images = []
        labels = []

        for i in ix:
            img, _ = self.load_image(imageDataset[i])
            images.extend(img)
            labels.extend(embedingsDataset[i])

        images = np.reshape(images, (n_samples, self.img_shape[0], self.img_shape[1], self.img_shape[2]))
        labels = np.reshape(labels, (n_samples, self.text_shape,))
        y = ones((n_samples, 1))

        return images, labels, y