#from StageOneDiscriminator import StageOneDiscriminator, define_discriminator_stage_one
#from StageOneGenerator import StageOneGenerator, define_generator_stage_one
from glove_loader import GloveModel


#from StageOneGAN import StageOneGan, define_gan_stage_one
from StageOneGAN import StageOneGAN

from PIL import Image
import tensorflow as tf
import numpy as np
import os
import json
import random
import collections
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.models import load_model
from load_birds import load_birds


from numpy import asarray
from numpy import savez_compressed
from numpy import load

import fiftyone as fo



def load_image(image_path, img_shape=(64,64,3)):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=img_shape[2])
        img = tf.keras.layers.Resizing(img_shape[0], img_shape[1])(img)
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img


def main():
    #IDEAS:
    #1. Change discriminator to tanh activation
    #        1 - Proper image
    #        0 - Mismatched label
    #       -1 - Wrong image wrong
    #2. Go back to only 0 and 1
    gan_name = "coco"
    img_shape = (64,64,3)
    paths_limit = 50
    #seems to work on 200
    #testing 300
    text_size = 200
    latent_dim = 100
    epochs = 1000
    batch = 64
    batch_size = int(paths_limit/5)
    dataset = []
    train_dataset_file = 'dataset_%d.npy' % paths_limit


    print("Dataset shape:")
    print(np.shape(dataset))
    if(np.shape(dataset)[0] == 0):
        print("Dataset is empty!")
        return
    split = int(0.9 * np.shape(dataset)[0])
    train_dataset = dataset[:split]
    val_dataset = dataset[split:]
    random.shuffle(train_dataset)
    random.shuffle(val_dataset)
    print("Training dataset shape:")
    print(np.shape(train_dataset))
    print("Validation dataset shape:")
    print(np.shape(val_dataset))


    ganOne = StageOneGAN(train_dataset, val_dataset, text_shape=text_size, latent_dim=latent_dim, image_shape=img_shape)
    #g = load_model("models/generator_models_coco/generator_model_1000.h5")
    plt.imshow


    ganOne.init_model()
    ganOne.name = gan_name
    ganOne.train_two(n_epochs=epochs, n_batch=batch)




if __name__ == "__main__":
    main()