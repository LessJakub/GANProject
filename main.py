from glove_loader import GloveModel

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



def load_image(image_path, img_shape=(64,64,3)):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=img_shape[2])
        img = tf.keras.layers.Resizing(img_shape[0], img_shape[1])(img)
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img

#antialias seems to not work properly for discriminator
def load_image_two(image_path, img_shape=(64,64,3)):
        img = Image.open(image_path)
        img = img.resize((img_shape[0], img_shape[1]), Image.ANTIALIAS)
        img = np.array(img)
        return img

def main():
    gan_name = "coco"
    img_shape = (64,64,3)
    paths_limit = 50
    text_size = 200
    latent_dim = 100
    epochs = 1000
    batch = 64
    batch_size = int(paths_limit/5)
    dataset = []
    train_dataset_file = 'dataset_%d.npy' % paths_limit

    if gan_name == "coco" and not os.path.exists(os.path.abspath('.') + "/" + train_dataset_file):
        annotation_file = ""
        annotation_folder = '/annotations/'
        if not os.path.exists(os.path.abspath('.') + annotation_folder):
            annotation_zip = tf.keras.utils.get_file('captions.zip',
                                           cache_subdir=os.path.abspath('.'),
                                           origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                           extract=True)
            annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
            os.remove(annotation_zip)

        annotation_file = os.getcwd() +'/annotations/captions_train2014.json'

        # Download image files
        image_folder = '/train2014/'
        if not os.path.exists(os.path.abspath('.') + image_folder):
            image_zip = tf.keras.utils.get_file('train2014.zip',
                                      cache_subdir=os.path.abspath('.'),
                                      origin='http://images.cocodataset.org/zips/train2014.zip',
                                      extract=True)
            PATH = os.path.dirname(image_zip) + image_folder
            os.remove(image_zip)
        else:
            PATH = os.path.abspath('.') + image_folder


        # Group all captions together having the same image ID.
        image_path_to_caption = collections.defaultdict(list)
    
        with open(annotation_file, 'r') as f:
            annotations = json.loads(f.read())

        for val in annotations['annotations']:
            caption = f"<start> {val['caption']} <end>"
            image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
            caption = caption.removeprefix("<start>").removesuffix("<end>")
            image_path_to_caption[image_path].append(caption)

        image_paths = list(image_path_to_caption.keys())
        random.shuffle(image_paths)


        image_paths = image_paths[:paths_limit]
        batches = int(len(image_paths) / batch_size)
        for i in tqdm(range(batches)):
            batch_start = i * batch_size
            batch_end = batch_start + batch_size
            for image_path in tqdm(image_paths[batch_start:batch_end]):
                caption_list = image_path_to_caption[image_path]
                for c in caption_list:
                    img = load_image(image_path, img_shape)
                    tmp = np.array((img, c))
                    dataset.append(tmp)
            with open(train_dataset_file, 'wb') as f:
                np.save(f, dataset)
    
        with open(train_dataset_file, 'rb') as f:
            dataset = np.load(f, allow_pickle=True)
    else:
        with open(train_dataset_file, 'rb') as f:
            dataset = np.load(f, allow_pickle=True)

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
    plt.imshow


    ganOne.init_model()
    ganOne.name = gan_name
    ganOne.train_two(n_epochs=epochs, n_batch=batch)


if __name__ == "__main__":
    main()