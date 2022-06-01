from StageOneGAN import StageOneGAN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Lambda, Embedding, Flatten, Conv1D
from keras.models import load_model
import os
from textwrap import wrap
import random
import numpy as np

import StageOneDiscriminator as d
import StageOneGenerator as g
from StageTwoGenerator import StageTwoGenerator
from StageTwoDiscriminator import StageTwoDiscriminator
from glove_loader import GloveModel
import matplotlib.pyplot as plt
from numpy.random import randint


class StageTwoGAN(object):
    def __init__(self, train_dataset, val_dataset, name ="", text_shape = 100, latent_dim = 100, image_shape = (256,256,3)):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.text_shape = text_shape
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.name = name

        self.glove = GloveModel()
        text_input_dim = text_shape
        glove_source_dir_path = './very_large_data'
        print("Loading glove")
        self.glove.load(data_dir_path=glove_source_dir_path, embedding_dim=text_input_dim)


    def init_model(self, g_one = None, d_one = None):
        #self.stageOneGan = StageOneGAN(,text_shape, latent_dim, (int(image_shape[0]/4), int(image_shape[1]/4), int(image_shape[2])))
        if(g_one == None):
            print("Stage one generator not provided!")
        else:
            self.g_one = g_one
        if(d_one == None):
            print("Stage one discriminator not provided!")
        else:
            self.d_one = d_one

        inputText = Input(shape = self.text_shape)
        inputNoise = Input(shape= (self.latent_dim,))

        g_one_output = self.g_one((inputNoise, inputText))
        #d_one_output = self.d_one((g_one_output, inputText))

        self.Generator = StageTwoGenerator((64,64,3), self.text_shape, self.latent_dim)
        #self.Generator.model = load_model("generator_models/generator_model_750.h5")
        self.Discriminator = StageTwoDiscriminator(self.image_shape, self.text_shape)


        g_two_output = self.Generator.model((g_one_output, inputText))
        d_two_output = self.Discriminator.model((g_two_output, inputText))

        self.model = Model([inputNoise, inputText], outputs=d_two_output)

        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        self.model.compile(loss='binary_crossentropy', optimizer=opt)
        from keras.utils.vis_utils import plot_model

        plot_model(self.Generator.model, to_file='g_two_plot.png', show_shapes=True, show_layer_names=True)
        plot_model(self.Discriminator.model, to_file='d_two_plot.png', show_shapes=True, show_layer_names=True)
        plot_model(self.model, to_file='gan_two_plot.png', show_shapes=True, show_layer_names=True)

    # # train the generator and discriminator
    def train_two(self, n_epochs=200, n_batch=128):
        bat_per_epo = int(self.train_dataset.shape[0] / n_batch)
        half_batch = int(n_batch / 2)
        quarter_batch = int(n_batch / 4)
        epoch_gen_losses = []
        epoch_dis_losses_fake = []
        epoch_dis_losses_real = []
        dis_acc_real = []
        dis_acc_fake = []
	    # manually enumerate epochs
        for i in range(n_epochs):
	    # enumerate batches over the training set
            for j in range(bat_per_epo):
                start = j * n_batch
                end = start + n_batch
                if(end > self.train_dataset.shape[0]):
                    end = self.train_dataset.shape[0]
			    # get randomly selected 'real' samples
                #i_real = randint(start, end, half_batch)
                i_real = random.sample(range(start, end), n_batch)
                x_real = []
                l_real = []
                for a in self.train_dataset[i_real]:
                    x_real.append(a[0])
                    l_real.append(self.glove.encode_doc(a[1]))
                x_real = np.array(x_real)
                l_real = np.array(l_real)
                y_real = np.ones((n_batch, 1))
                
			    # update discriminator model weights
                self.Discriminator.model.trainable = True
                d_loss1, _ = self.Discriminator.model.train_on_batch((x_real, l_real), y_real)



			    # generate 'fake' examples
                #i_fake = randint(start, end, half_batch)
                i_fake = random.sample(range(start, end), half_batch)
                lat_points = self.Generator.generate_latent_points(half_batch)
                l_fake = []
                for b in self.train_dataset[i_fake]:
                    l_fake.append(self.glove.encode_doc(b[1]))
                l_fake = np.array(l_fake)
                x_one = self.g_one.predict((lat_points, l_fake))
                x_fake = self.Generator.model.predict((x_one, l_fake))
                y_fake = np.zeros((half_batch, 1))
			    # update discriminator model weights
                d_loss2, _ = self.Discriminator.model.train_on_batch((x_fake, l_fake), y_fake)
			    # prepare points in latent space as input for the generator

                #Generator creates image replicas a lot faster but some labels are mismatched
                i_mis = random.sample(range(start, end), half_batch)
                x_mis = []
                l_mis = []
                for c in i_mis:
                    x_mis.append(self.train_dataset[c][0])
                    of = 1
                    while(c + of < len(self.train_dataset) and np.array_equal(self.train_dataset[c][0], self.train_dataset[c + of][0])):
                        of += 1
                    l_mis.append(self.glove.encode_doc(self.train_dataset[c + of][1]))
                x_mis = np.array(x_mis)
                l_mis = np.array(l_mis)
                y_mis = np.zeros((half_batch, 1))
                
                d_loss3, _ = self.Discriminator.model.train_on_batch((x_mis, l_mis), y_mis)



                self.Discriminator.model.trainable = False


                #i_gan = randint(start, end, n_batch)
                i_gan = random.sample(range(start, end), n_batch)
                lat = self.Generator.generate_latent_points(n_batch)
                l_gan = []
                for d in self.train_dataset[i_gan]:
                    l_gan.append(self.glove.encode_doc(d[1]))
                #x_gan = np.array(x_gan)
                l_gan = np.array(l_gan)
			    # create inverted labels for the fake samples
                y_gan = np.ones((n_batch, 1))
                #x_gan = self.g_one.predict((lat, l_gan))
			    # update the generator via the discriminator's error
                g_loss = self.model.train_on_batch((lat, l_gan), y_gan)
			    # summarize loss on this batch
                print('>%d, %d/%d, d1=%.3f, d2=%.3f, d3=%.3f, g=%.3f' %
			    	(i+1, j+1, bat_per_epo, d_loss1, d_loss2, d_loss3, g_loss))
                epoch_gen_losses.append(g_loss)
                epoch_dis_losses_real.append(d_loss1)
                epoch_dis_losses_fake.append(d_loss2)
		    # evaluate the model performance, sometimes
            n = 2
            samples = n * n
            i_real = randint(0, self.val_dataset.shape[0], samples)
            x_real = []
            l_real = []
            for e in self.val_dataset[i_real]:
                x_real.append(e[0])
                l_real.append(self.glove.encode_doc(e[1]))
                
            x_real = np.array(x_real)
            l_real = np.array(l_real)
            y_real = np.ones((samples, 1))
            _, acc_real = self.Discriminator.model.evaluate((x_real, l_real), y_real, verbose=0)

            i_fake = randint(0, self.val_dataset.shape[0], samples)
            lat_points = self.Generator.generate_latent_points(samples)
            l_fake = []
            for f in self.val_dataset[i_fake]:
                l_fake.append(self.glove.encode_doc(f[1]))
            l_fake = np.array(l_fake)
            x_one = self.g_one.predict((lat_points, l_fake))
            x_fake = self.Generator.model.predict((x_one, l_fake))
            x_fake = np.array(x_fake)
            y_fake = np.zeros((samples, 1))
            #for l in l_fake:
            #    l = self.glove.encode_doc(l)
            _, acc_fake = self.Discriminator.model.evaluate((x_fake, l_fake), y_fake, verbose=0)


            dis_acc_real.append(acc_real)
            dis_acc_fake.append(acc_fake)
            acc_info = '>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100)
            print(acc_info)

            

            if((i + 1) % 10 == 0):
                path =  "models/generator_models" + "_" + self.name + "/"
                filename = path + 'generator_model_%04d.h5' % (i+1)
                self.Generator.model.save(filename)

                path = "models/discriminator_models" + "_" + self.name + "/"
                filename = path + 'discriminator_model_%04d.h5' % (i+1)
                self.Discriminator.model.save(filename)

                i_g_v = randint(0, self.val_dataset.shape[0], int(samples/2))
                i_g_t = randint(0, self.train_dataset.shape[0], int(samples/2))
                lat = self.Generator.generate_latent_points(samples)
                
                #l_c = np.array(self.val_dataset[i_g][1])
                #split = int(len(i_g)/2)
                l_c = []
                for e in self.val_dataset[i_g_v]:
                    l_c.append(e[1])
                for e in self.train_dataset[i_g_t]:
                    l_c.append(e[1])
                l_g = []
                for z in range(len(l_c)):
                    l_g.append(self.glove.encode_doc(l_c[z]))
               
                l_c = np.array(l_c)
                l_g = np.array(l_g)
                x_g = self.g_one.predict((lat, l_g))
                x_g = np.array(x_g)
                imgs = self.Generator.model.predict((x_g, l_g))
                imgs = (imgs + 1) / 2.0
                my_dpi = 80
                fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
                #fig.title.set_text(acc_info)
                for y in range(n):
                    for x in range(n):
                        axes[y,x].axis('off')
                        axes[y,x].imshow(imgs[n*y + x])
                        #tmp = ' '.join([tf.compat.as_text(index_to_word(i).numpy())
                        #     for i in labels.tolist()[n*y + x] if i not in [0]])

                        axes[y,x].title.set_text("\n".join(wrap("{}. {}".format(n*y + x, l_c[n*y + x]) , 25)))
             
                plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.8, 
                        wspace=0.6, 
                        hspace=0.6)

                path = "images/images" + "_" + self.name
                if not os.path.exists(os.path.abspath('.') + "/" + path):
                    os.mkdir(path)
                filename = path + '/generated_plot_e%04d.png' % (i + 1)
                plt.savefig(filename)
                plt.close()
            

                plt.plot(epoch_gen_losses, label = "gen")
                plt.plot(epoch_dis_losses_real, label = "dis_real")
                plt.plot(epoch_dis_losses_fake, label = "dis_fake")
                plt.legend(loc="upper right")

                path = "plots/plots_{}/".format(self.name)
                if not os.path.exists(os.path.abspath('.') + "/" + path):
                    os.mkdir(os.path.abspath('.') + "/" + path)

                path = "plots/plots_{}/loss".format(self.name)
                if not os.path.exists(os.path.abspath('.') + "/" + path):
                    os.mkdir(path + "/")
                plt.savefig(path + "/epoch_{}.png".format( i + 1))
                plt.close()

                plt.plot(dis_acc_real, label = "real")
                plt.plot(dis_acc_fake, label = "fake")
                plt.legend(loc="upper right")

                path = "plots/plots_{}/acc".format(self.name)
                if not os.path.exists(os.path.abspath('.') + "/" + path):
                    os.mkdir(os.path.abspath('.') + "/" + path + "/")
                plt.savefig(path + "/epoch_{}.png".format( i + 1))
                plt.close()