from collections import defaultdict
from email.policy import default
import pickle
import random
import numpy as np
import pandas as pd
import PIL
import os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf


def load_image(image_path, img_shape=(64,64,3)):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=img_shape[2])
        img = tf.keras.layers.Resizing(img_shape[0], img_shape[1])(img)
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img

#Load class IDs
def load_class_ids(class_info_file_path):
    """
    Load class ids from class_info.pickle file
    """
    with open(class_info_file_path, 'rb') as f:
        class_ids = pickle.load(f, encoding='latin1')
        return class_ids

def load_filenames(filenames_file_path):
    """
    Load filenames.pickle file and return a list of all file names
    """
    with open(filenames_file_path, 'rb') as f:
        filenames = pickle.load(f, encoding='latin1')
    return filenames

def load_embeddings(embeddings_file_path):
    """
    Load embeddings
    """
    with open(embeddings_file_path, 'rb') as f:
        embeddings = pickle.load(f, encoding='latin1')
        embeddings = np.array(embeddings)
        print('embeddings: ', embeddings.shape)
    return embeddings

def load_bounding_boxes(dataset_dir):
    """
    Load bounding boxes and return a dictionary of file names and corresponding bounding boxes
    """
    # Paths
    bounding_boxes_path = os.path.join(dataset_dir, 'bounding_boxes.txt')
    file_paths_path = os.path.join(dataset_dir, 'images.txt')

    # Read bounding_boxes.txt and images.txt file
    df_bounding_boxes = pd.read_csv(bounding_boxes_path,
                                    delim_whitespace=True, header=None).astype(int)
    df_file_names = pd.read_csv(file_paths_path, delim_whitespace=True, header=None)

    # Create a list of file names
    file_names = df_file_names[1].tolist()

    # Create a dictionary of file_names and bounding boxes
    filename_boundingbox_dict = {img_file[:-4]: [] for img_file in file_names[:2]}

    # Assign a bounding box to the corresponding image
    for i in range(0, len(file_names)):
        # Get the bounding box
        bounding_box = df_bounding_boxes.iloc[i][1:].tolist()
        key = file_names[i][:-4]
        filename_boundingbox_dict[key] = bounding_box

    return filename_boundingbox_dict


def get_img(img_path, bbox, image_size):
    """
    Load and resize image
    """
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - R)
        y2 = np.minimum(height, center_y + R)
        x1 = np.maximum(0, center_x - R)
        x2 = np.minimum(width, center_x + R)
        img = img.crop([x1, y1, x2, y2])
    img = img.resize(image_size, PIL.Image.BILINEAR)
    return img

def get_text(img_filename, text_dir):
    #print(img_filename)
    #for f in os.listdir():
    #    print(f)
    with open(text_dir + "/" + img_filename + ".txt") as f:
        descriptions = f.readlines()
        for d in descriptions:
            d = d.removesuffix("\n")
    #print(descriptions)
    return descriptions

def load_dataset(filenames_file_path, class_info_file_path, cub_dataset_dir, embeddings_file_path, image_size, text_file_path):
    filenames = load_filenames(filenames_file_path)
    class_ids = load_class_ids(class_info_file_path)
    bounding_boxes = load_bounding_boxes(cub_dataset_dir)
    all_embeddings = load_embeddings(embeddings_file_path)

    X, y, embeddings, descs, f, d = [], [], [], [], [], []
    dictonary = defaultdict()

    # TODO: Change filenames indexing
    for index, filename in enumerate(filenames[:500]):
        # print(class_ids[index], filenames[index])
        bounding_box = bounding_boxes[filename]

        try:
            # Load images
            img_name = '{}/images/{}.jpg'.format(cub_dataset_dir, filename)
            #img = get_img(img_name, bounding_box, image_size)
            img = load_image(img_name)
            t = get_text(filename, text_file_path)
            for l in t:
                tmp = np.array((img, l))
                d.append(tmp)
            all_embeddings1 = all_embeddings[index, :, :]

            embedding_ix = random.randint(0, all_embeddings1.shape[0] - 1)
            embedding = all_embeddings1[embedding_ix, :]

            dictonary[filename] = (np.array(img), t)
            X.append(np.array(img))
            y.append(class_ids[index])
            f.append(filename)
            embeddings.append(embedding)
            descs.append(t)
        except Exception as e:
            print(e)

    X = np.array(X)
    y = np.array(y)
    embeddings = np.array(embeddings)
    
    return d


def load_birds():
    #Hyperparameters
    data_dir = "C:\\repos\\OReillySGAN\\birds"
    train_dir = data_dir + "/train"
    test_dir = data_dir + "/test"
    image_size = 64
    batch_size = 64
    z_dim = 100
    stage1_generator_lr = 0.0002
    stage1_discriminator_lr = 0.0002
    stage1_lr_decay_step = 600
    epochs = 2000
    condition_dim = 128

    embeddings_file_path_train = train_dir + "/char-CNN-RNN-embeddings.pickle"
    embeddings_file_path_test = test_dir + "/char-CNN-RNN-embeddings.pickle"

    filenames_file_path_train = train_dir + "/filenames.pickle"
    filenames_file_path_test = test_dir + "/filenames.pickle"

    class_info_file_path_train = train_dir + "/class_info.pickle"
    class_info_file_path_test = test_dir + "/class_info.pickle"
    text_file_path = data_dir + "/text_c10"
    cub_dataset_dir = data_dir + "/CUB_200_2011"

    dic = load_dataset(filenames_file_path=filenames_file_path_train,
                                                class_info_file_path=class_info_file_path_train,
                                                  cub_dataset_dir=cub_dataset_dir,
                                                embeddings_file_path=embeddings_file_path_train,
                                                text_file_path=text_file_path,
                                                  image_size=(64, 64))
    return np.array(dic)


if __name__ == "__main__":
    dic = load_birds()
    plt.imshow(dic[0][0])
    plt.title(dic[0][1])
    plt.show()