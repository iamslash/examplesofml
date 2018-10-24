import numpy as np
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt

class ElapsedTimer(object):
    def __init__(self):
        self.start = time.time()
    def elapsed(self, sec):
        if sec < 60:
            return "{} sec".format(sec)
        elif sec < 3600:
            return "{} min".format(sec / 60)
        else:
            return "{} hr".format(sec / 3600)
    def elapsed_time(self):
        print("Elapsed: {} ".format(self.elapsed(time.time() - self.start)))

class DcGan(object):
    def __init__(self, row=28, col=28, chn=1):
        self.img_row = row
        self.img_col = col
        self.img_chn = chn
        self.latent_size = 64
        self.hidden_size = 256
        self.img_size = row * col * chn

        self.D  = self.build_discriminator()       # discriminator
        self.G  = self.build_generator()       # generator       
        self.DL = None      # discriminator loss
        self.GL = None      # generator loss

    def build_discriminator(self):
        seq = Sequential()
        seq.add(Dense(self.hidden_size, input_dim=self.img_size))
        seq.add(LeakyReLU(alpha=0.2))
        seq.add(Dense(self.hidden_size, input_dim=self.hidden_size))
        seq.add(LeakyReLU(alpha=0.2))
        seq.add(Dense(1, input_dim=self.hidden_size))
        seq.add(Activation('sigmoid'))    
        seq.summary()    
        return seq

    def build_generator(self):
        seq = Sequential()
        seq.add(Dense(self.hidden_size, input_dim=self.latent_size))
        seq.add(LeakyReLU())
        seq.add(Dense(self.hidden_size, input_dim=self.hidden_size))
        seq.add(LeakyReLU())
        seq.add(Dense(self.img_size, input_dim=self.hidden_size))
        seq.add(Activation('tanh')) 
        seq.summary()       
        return seq

class MnistDcGan(object):
    def __init__(self):
        self.dcgan = DcGan()

        self.x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
        self.x_train = self.x_train.reshape(-1, self.dcgan.img_size).astype(np.float32)
        


    def train(self, epoch_size=200, batch_size=128, save_interval=1):

        for i in range(epoch_size):
            imgs = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :]
            real_labels = np.ones([batch_size, 1])
            fake_labels = np.zeros([batch_size, 1])

            # ============================================================ #
            #                    Train the discriminator                   #
            # ============================================================ #

            # ============================================================ #
            #                    Train the generator                       #
            # ============================================================ #

    def save_image(self, images, path):
        pass

if __name__ == '__main__':
    mdg = MnistDcGan()
    timer = ElapsedTimer()
    mdg.train(epoch_size=1, batch_size=128, save_interval=10)
    timer.elapsed_time()    