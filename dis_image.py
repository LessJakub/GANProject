from keras.models import load_model
from glove_loader import GloveModel
import matplotlib.pyplot as plt
from numpy.random import randn
import sys
import numpy as np

def main():
    #if(len(sys.argv) < 2):
    #    print("Please provide at least one description")
    #    return

    d_model = load_model('models/discriminator_models_coco/discriminator_model_0650.h5')
    #g_model.summary()
    dataset = np.load("dataset_500.npy", allow_pickle=True)
    n = len(sys.argv) - 1
    
    glove = GloveModel()
    text_input_dim = 100
    glove_source_dir_path = './very_large_data'
    print("Loading glove")
    glove.load(data_dir_path=glove_source_dir_path, embedding_dim=text_input_dim)

    x = []
    x.append(dataset[0][0])
    y = []
    y.append(glove.encode_doc(dataset[0][1]))
    z = []
    z.append(glove.encode_doc(""))

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    d_data = d_model.predict((x, y))
    d_fake = d_model.predict((x, z))

    print("From data: {}, from fake: {}".format(d_data * 100, d_fake * 100))
    plt.imshow(dataset[0][0])
    plt.title(dataset[0][1])
    plt.show()

if __name__ == "__main__":
    main()