import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.layers import ReLU
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate
from keras.layers import Dense
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.layers import Conv2DTranspose
from keras.layers import Reshape
from keras.layers import Embedding, UpSampling2D
from keras.layers import Concatenate, Lambda, Activation
from keras import backend as K
from tensorflow.keras.optimizers import SGD

from numpy.random import randn
from numpy.random import randint
from numpy import zeros


class StageOneGenerator(object):
#def define_generator_stage_one(voc_size, max_length, latent_dim=100):
    def __init__(self, latent_dim=100, text_shape=100):
        self.latent_dim = latent_dim
        self.text_shape = text_shape
        inputNoise = Input(shape = (latent_dim,))
        noise_dense = Dense(1024)(inputNoise)


        inputText = Input(shape = (text_shape,))
        text_dense = Dense(4*4*3)(inputText)
        combined = concatenate([noise_dense, text_dense])
        z = Activation('tanh')(combined)

        z = Dense(4*4*128*2)(z)
        z = BatchNormalization()(z)
        z = Activation('tanh')(z)
        z = Reshape((4,4,256))(z)

        z = UpSampling2D(size=(2,2))(z)
        z = Conv2D(256, kernel_size=5, padding='same')(z)
        z = Activation('tanh')(z)
        #8 x 8

        z = UpSampling2D(size=(2,2))(z)
        z = Conv2D(256, kernel_size=5, padding='same')(z)
        z = Activation('tanh')(z)
        #16 x 16

        z = UpSampling2D(size=(2,2))(z)
        z = Conv2D(256, kernel_size=5, padding='same')(z)
        z = Activation('tanh')(z)
        #32 x 32

        z = UpSampling2D(size=(2,2))(z)
        z = Conv2D(3, kernel_size=5, padding='same')(z)
        z = Activation('tanh')(z)
        #64 x 64

        #z = UpSampling2D(size=(2,2))(z)
        #z = Conv2D(3, kernel_size=5, padding='same')(z)
        #z = Activation('tanh')(z)
        #128 x 128

        #z = UpSampling2D(size=(2,2))(z)
        #z = Conv2D(3, kernel_size=5, padding='same')(z)
        #z = Activation('tanh')(z)
        #256 x 256
        
	    # # upsample to 8x8
        # z = Conv2DTranspose(64 * 8, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False)(combined)
        # z = BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02)(z)
        # z = ReLU()(z)
	    # # upsample to 16x16
        # z = Conv2DTranspose(64 * 4, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False)(z)
        # z = BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02)(z)
        # z = ReLU()(z)
	    # # upsample to 32x32
        # z = Conv2DTranspose(64 * 2, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False)(z)
        # z = BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02)(z)
        # z = ReLU()(z)
	    # upsample to 64x64
        #z = Conv2DTranspose(256, (4,4), strides=(1,1), padding='same')(z)
        #z = BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02)(z)
        #z = ReLU()(z)

        #z = Conv2DTranspose(3, 4, 2,padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False, activation='tanh')(z)
	    # output layer
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        self.model = Model([inputNoise, inputText], outputs=z)
        #self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['accuracy'])
        #return model

    def generate_latent_points(self, n_samples):
        # generate points in the latent space
        x_input = randn(self.latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape((n_samples, self.latent_dim,))
        return x_input

    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, embedingsDataset, n_samples):
        # generate points in latent space
        x_input = self.generate_latent_points(n_samples)

        ix = randint(0, embedingsDataset.shape[0], n_samples)

        l_input = embedingsDataset[ix]
        #l_input = ca(l_input)
        # predict outputs
        X = self.model.predict((x_input, l_input))
        # create 'fake' class labels (0)
        y = zeros((n_samples, 1))

        return X, l_input, y, ix


