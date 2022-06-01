from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate
from keras.layers import Dense, Activation, UpSampling2D, MaxPooling2D
from keras.layers import BatchNormalization, Reshape, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from numpy.random import randint
from numpy.random import randn
from numpy import zeros

class StageTwoGenerator(object):
    def __init__(self, img_shape=(64,64,3), text_shape=100, latent_dim = 100):
        self.img_shape = img_shape
        self.text_shape = text_shape
        self.latent_dim = latent_dim
        inputImage = Input(shape = img_shape)
        #noise_dense = Dense(1024)(inputNoise)

        x = Conv2D(64, kernel_size=(5,5), padding='same')(inputImage)
        x = Activation('tanh')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Conv2D(128, kernel_size=5)(x)
        x = Activation('tanh')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Conv2D(256, kernel_size=5)(x)
        x = Activation('tanh')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        
        
        x = Flatten()(x)
        x = Dense(1024)(x)

        inputText = Input(shape = (text_shape,))
        text_dense = Dense(1024)(inputText)
        combined = concatenate([x, text_dense])
        z = Activation('tanh')(combined)

        z = Dense(4*4*128*2*2)(z)
        z = BatchNormalization()(z)
        z = Activation('tanh')(z)
        z = Reshape((4,4,512))(z)

        z = UpSampling2D(size=(2,2))(z)
        z = Conv2D(256, kernel_size=5, padding='same')(z)
        z = Activation('tanh')(z)
        z = UpSampling2D(size=(2,2))(z)
        z = Conv2D(256, kernel_size=5, padding='same')(z)
        z = Activation('tanh')(z)

        z = UpSampling2D(size=(2,2))(z)
        z = Conv2D(512, kernel_size=5, padding='same')(z)
        z = Activation('tanh')(z)
        z = UpSampling2D(size=(2,2))(z)
        z = Conv2D(512, kernel_size=5, padding='same')(z)
        z = Activation('tanh')(z)

        z = UpSampling2D(size=(2,2))(z)
        z = Conv2D(256, kernel_size=5, padding='same')(z)
        z = Activation('tanh')(z)
        z = UpSampling2D(size=(2,2))(z)
        z = Conv2D(3, kernel_size=5, padding='same')(z)
        z = Activation('tanh')(z)

        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        self.model = Model([inputImage, inputText], outputs=z, name='GeneratorTwo')
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        #self.model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['accuracy'])

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

    def generate_latent_points(self, n_samples):
        # generate points in the latent space
        x_input = randn(self.latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, self.latent_dim)
        return x_input
