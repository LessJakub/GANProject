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
#def define_discriminator_stage_one(voc_size, max_length, image_shape=(64,64,3)):
    def __init__(self, img_shape=(64,64,3), text_shape=100):
        self.img_shape = img_shape
        self.text_shape = text_shape
        inputText = Input(shape=(text_shape,))
        #change to 64*64*3 and hope for no OOM
        #for 1024 it works but guesses everything as 0
        text_dense = Dense(512)(inputText)
        #test next: text_pooling = MaxPooling1D(pool_size=2)(inputText)
        #test next: text_dense = Dense(1024)(text_pooling)
   

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
        #return model

    def train_discriminator(self,imgDataset, embDataset, n_iter=20, n_batch=128):
        half_batch = int(n_batch / 2)
        # manually enumerate epochs
        for i in range(n_iter):
		    # get randomly selected 'real' samples
            X_real, y_real, l_real = generate_real_samples(imgDataset, embDataset, half_batch)
		    # update discriminator on real samples
            _, real_acc = self.model.train_on_batch((X_real, l_real), y_real)
		    # generate 'fake' examples
            X_fake, y_fake, l_fake = generate_real_samples(imgDataset, embDataset, half_batch)
		    # update discriminator on fake samples
            _, fake_acc = self.model.train_on_batch((X_fake, l_fake), y_fake)
            # summarize performance
            print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))


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