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
import fiftyone.zoo as foz
from fiftyone import ViewField as F


def load_image(image_path, img_shape=(64,64,3)):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=img_shape[2])
        img = tf.keras.layers.Resizing(img_shape[0], img_shape[1])(img)
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img


def main():
    dataset = foz.load_zoo_dataset("quickstart")
    print(dataset)

    # Evaluate the objects in the `predictions` field with respect to the
    # objects in the `ground_truth` field
    results = dataset.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        method="coco",
        eval_key="eval",
    )

    # Get the 10 most common classes in the dataset
    counts = dataset.count_values("ground_truth.detections.label")
    classes = sorted(counts, key=counts.get, reverse=True)[:10]

    # Print a classification report for the top-10 classes
    results.print_report(classes=classes)

    # Print some statistics about the total TP/FP/FN counts
    print("TP: %d" % dataset.sum("eval_tp"))
    print("FP: %d" % dataset.sum("eval_fp"))
    print("FN: %d" % dataset.sum("eval_fn"))

    # Create a view that has samples with the most false positives first, and
    # only includes false positive boxes in the `predictions` field
    view = (
        dataset
        .sort_by("eval_fp", reverse=True)
        .filter_labels("predictions", F("eval") == "fp")
    )

    # Visualize results in the App
    session = fo.launch_app(view=view)

    gan_name = "coco"
    img_shape = (64,64,3)
    paths_limit = 50
    text_size = 200
    latent_dim = 100
    epochs = 1000
    batch = 64
    batch_size = int(paths_limit/5)

    return
    ganOne = StageOneGAN(train_dataset, val_dataset, text_shape=text_size, latent_dim=latent_dim, image_shape=img_shape)
    #g = load_model("models/generator_models_coco/generator_model_1000.h5")
    plt.imshow


    ganOne.init_model()
    ganOne.name = gan_name
    ganOne.train_two(n_epochs=epochs, n_batch=batch)





if __name__ == "__main__":
    main()