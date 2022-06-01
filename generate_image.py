from keras.models import load_model
from glove_loader import GloveModel
import matplotlib.pyplot as plt
from numpy.random import randn
import sys
import numpy as np

def main():
    if(len(sys.argv) < 2):
        print("Please provide at least one description")
        return

    g_model = load_model('models/generator_models_coco/generator_model_0500.h5')
    #g_model.summary()

    n = len(sys.argv) - 1
    n = 4
    samples = n * n

    tmp_token = []
    
    glove = GloveModel()
    text_input_dim = 200
    glove_source_dir_path = './very_large_data'
    print("Loading glove")
    glove.load(data_dir_path=glove_source_dir_path, embedding_dim=text_input_dim)

    
    x_lat = randn(100 * samples)
    x_lat = x_lat.reshape((samples, 100))
    for i in range(samples):
        tmp_token.append(glove.encode_doc(sys.argv[1]))

    #for i in range(1, n + 1):
    #    print(sys.argv[i])
    #    tmp_token.append(glove.encode_doc(sys.argv[i]))
    #    #print(tmp_token[i - 1])

    tmp_token = np.array(tmp_token)
    predicts = g_model.predict((x_lat, tmp_token))
    predicts = (predicts + 1)/2
    my_dpi = 70


    #plt.imshow(predicts[0])
    #plt.title(sys.argv[1])
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
    for y in range(n):
        for x in range(n):
            axes[y,x].axis('off')
            axes[y,x].imshow(predicts[n*y + x])
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()