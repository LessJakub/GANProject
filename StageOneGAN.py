from re import X
import StageOneDiscriminator as d
import StageOneGenerator as g

import tensorflow as tf
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Lambda, Embedding, Flatten, Conv1D
from IPython import display
import telegram
from tqdm import trange
from numpy.random import randint
from textwrap import wrap
from keras.models import load_model
from glove_loader import GloveModel
from keras.utils.vis_utils import plot_model

import os
import numpy as np
import time

import matplotlib.pyplot as plt

#bot = telegram.Bot("5350881613:AAH9sxMdK3MDHZL_0YeP0P8PkOOHdjFF5D8")


#def define_gan_stage_one(d_model, g_model, ca, latent_dim=100, text_shape=(24,)):
class StageOneGAN(object):
    def __init__(self, dataset, val_dataset, text_shape = 100, latent_dim = 100, image_shape = (64,64,3), name=""):
        self.text_shape = text_shape
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.train_dataset = dataset
        self.val_dataset = val_dataset
        self.name = name
        self.glove = GloveModel()
        text_input_dim = text_shape
        glove_source_dir_path = './very_large_data'
        print("Loading glove")
        self.glove.load(data_dir_path=glove_source_dir_path, embedding_dim=text_input_dim)


    def generate_c(self, x):
        mean = x[:, :128]
        log_sigma = x[:, 128:]

        stddev = K.exp(log_sigma)
        epsilon = K.random_normal(shape=K.constant((mean.shape[1], ), dtype='int32'))
        c = stddev * epsilon + mean

        return c

    def build_ca_model(self,):
        input_layer = Input(shape=(self.text_shape,))
        x = Dense(256)(input_layer)
        #x = Embedding(100, 256)(input_layer)
        #x = Dense(256)(x)
        mean_logsigma = LeakyReLU(alpha=0.2)(x)

        c = Lambda(self.generate_c)(mean_logsigma)
        return Model(inputs=[input_layer], outputs=[c])

    def init_model(self, g_model = None, d_model = None):
        self.Generator = g.StageOneGenerator(self.latent_dim, self.text_shape)
        if(g_model != None):
            self.Generator.model = g_model
        
        self.Discriminator = d.StageOneDiscriminator(self.image_shape, self.text_shape)
        if(d_model != None):
            self.Discriminator.model = d_model

        inputText = Input(shape = self.text_shape)
        inputNoise = Input(shape= (self.latent_dim,))

        # make weights in the discriminator not trainable
        self.Discriminator.model.trainable = False
        # connect them
        

        x = self.Generator.model((inputNoise, inputText))
        
        x = self.Discriminator.model((x, inputText))

        self.model = Model([inputNoise, inputText], outputs=x)

        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        self.model.compile(loss='binary_crossentropy', optimizer=opt)
        
        plot_model(self.Generator.model, to_file='g_plot.png', show_shapes=True, show_layer_names=True)
        plot_model(self.Discriminator.model, to_file='d_plot.png', show_shapes=True, show_layer_names=True)
        plot_model(self.model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)
        #return model

        
    # # train the generator and discriminator
    def train_two(self, n_epochs=200, n_batch=128):
        bat_per_epo = int(self.train_dataset.shape[0] / n_batch)
        half_batch = int(n_batch / 2)
        epoch_gen_losses = []
        epoch_dis_losses_fake = []
        epoch_dis_losses_real = []
        dis_acc_real = []
        dis_acc_fake = []
	    # manually enumerate epochs
        for i in range(n_epochs):
	    # enumerate batches over the training set
            for j in range(bat_per_epo):
			    # get randomly selected 'real' samples
                i_real = randint(0, self.train_dataset.shape[0], half_batch)
                x_real = []
                l_real = []
                for e in self.train_dataset[i_real]:
                    x_real.append(e[0])
                    l_real.append(self.glove.encode_doc(e[1]))
                x_real = np.array(x_real)
                l_real = np.array(l_real)
                y_real = np.ones((half_batch, 1))
                
			    # update discriminator model weights
                self.Discriminator.model.trainable = True

                d_loss1, _ = self.Discriminator.model.train_on_batch((x_real, l_real), y_real)

			    # generate 'fake' examples
                i_fake = randint(0, self.train_dataset.shape[0], half_batch)
                lat_points = self.Generator.generate_latent_points(half_batch)
                l_fake = []
                for e in self.train_dataset[i_fake]:
                    l_fake.append(self.glove.encode_doc(e[1]))
                l_fake = np.array(l_fake)
                x_fake = self.Generator.model.predict((lat_points, l_fake))
                y_fake = np.zeros((half_batch, 1))
			    # update discriminator model weights
                d_loss2, _ = self.Discriminator.model.train_on_batch((x_fake, l_fake), y_fake)
			    # prepare points in latent space as input for the generator
                self.Discriminator.model.trainable = False


                i_gan = randint(0, self.train_dataset.shape[0], n_batch)
                x_gan = self.Generator.generate_latent_points(n_batch)
                l_gan = []
                for e in self.train_dataset[i_gan]:
                    l_gan.append(self.glove.encode_doc(e[1]))
                #x_gan = np.array(x_gan)
                l_gan = np.array(l_gan)
			    # create inverted labels for the fake samples
                y_gan = np.ones((n_batch, 1))
			    # update the generator via the discriminator's error
                g_loss = self.model.train_on_batch((x_gan, l_gan), y_gan)
			    # summarize loss on this batch
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
			    	(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
                epoch_gen_losses.append(g_loss)
                epoch_dis_losses_real.append(d_loss1)
                epoch_dis_losses_fake.append(d_loss2)
		    # evaluate the model performance, sometimes
            n = 4
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
            for e in self.val_dataset[i_fake]:
                l_fake.append(self.glove.encode_doc(e[1]))
            l_fake = np.array(l_fake)
            x_fake = self.Generator.model.predict((lat_points, l_fake))
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
                x_g = self.Generator.generate_latent_points(samples)
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
                x_g = np.array(x_g)
                l_c = np.array(l_c)
                l_g = np.array(l_g)
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
    

        #x_fake, epoch, acc_info, l_fake, labels, index_to_word
    def save_plot(self, examples, epoch, acc_info, labels, n=4):
	    # scale from [-1,1] to [0,1]
        examples = (examples + 1) / 2.0  
    
        my_dpi = 80
        fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
        #fig.set_size_pixels(480, 480)
    
        for y in range(n):
            for x in range(n):
                axes[y,x].axis('off')
                axes[y,x].imshow(examples[n*y + x])
                #tmp = ' '.join([tf.compat.as_text(index_to_word(i).numpy())
                #         for i in labels.tolist()[n*y + x] if i not in [0]])
                #tmp =""
                #axes[y,x].title.set_text("\n".join(wrap(tmp, 25)))
             
        #plt.subplots_adjust(left=0.1,
        #            bottom=0.1, 
        #            right=0.9, 
        #            top=0.8, 
        #            wspace=0.8, 
        #            hspace=0.8)


	    # save plot to file
        #plt.show()
        path = "images/"
        filename = path + 'generated_plot_e%04d.png' % (epoch+1)
        plt.savefig(filename)
        #bot.sendPhoto(chat_id="5386844483", caption='Epoch: %03d' % (epoch+1) + acc_info, photo=open(filename, 'rb'))
        plt.close()

    # evaluate the discriminator, plot generated images, save generator model
    def summarize_performance(self, imageDataset, embedingsDataset, epoch, n_samples=16):
	    # prepare real samples
        X_real, l_real, y_real = self.Discriminator.generate_real_samples(imageDataset, embedingsDataset, n_samples)
        # evaluate discriminator on real examples

        #l_real = ca(l_real)
        _, acc_real = self.Discriminator.model.evaluate((X_real, l_real), y_real, verbose=0)
        # prepare fake examples
        x_fake, l_fake, y_fake, ix = self.Generator.generate_fake_samples(embedingsDataset, n_samples)
        #l_fake = ca(l_fake)
        # evaluate discriminator on fake examples
        _, acc_fake = self.Discriminator.model.evaluate((x_fake, l_fake), y_fake, verbose=0)
        # summarize discriminator performance
        acc_info = '>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100)
        print(acc_info)
        # save plot
        labels = embedingsDataset[ix]
        self.save_plot(x_fake, epoch, acc_info, labels)
        #l = self.Generator.generate_latent_points(100, np.shape(labels)[0])
        #_, gen_loss = g_model.evaluate((l,labels))
        #print(gen_loss)
        # save the generator model tile file
        path = "generator_models/"
        filename = path + 'generator_model_%04d.h5' % (epoch+1)
        self.Generator.model.save(filename)
        #return 