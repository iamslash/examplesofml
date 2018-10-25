import os

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

IMGDIR = './imgs'

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
        self.DM = self.build_discriminator_model()      # discriminator loss
        self.AM = self.build_adversarial_model()      # generator loss

    def build_discriminator(self):
        seq = Sequential()
        seq.add(Dense(self.hidden_size, input_dim=self.img_size))
        seq.add(LeakyReLU(alpha=0.2))
        seq.add(Dense(self.hidden_size))
        seq.add(LeakyReLU(alpha=0.2))
        seq.add(Dense(1))
        seq.add(Activation('sigmoid'))    
        seq.summary()    
        return seq

    def build_generator(self):
        seq = Sequential()
        seq.add(Dense(self.hidden_size, input_dim=self.latent_size))
        seq.add(LeakyReLU())
        seq.add(Dense(self.hidden_size))
        seq.add(LeakyReLU())
        seq.add(Dense(self.img_size))
        seq.add(Activation('tanh')) 
        seq.summary()       
        return seq
    
    def build_discriminator_model(self):
        seq = Sequential()
        seq.add(self.D)
        seq.compile(optimizer=Adam(lr=0.0002), loss='binary_crossentropy', metrics=['accuracy'])
        return seq

    def build_adversarial_model(self):
        seq = Sequential()
        seq.add(self.G)
        seq.add(self.D)
        seq.compile(optimizer=Adam(lr=0.0002), loss='binary_crossentropy',metrics=['accuracy'])
        return seq

class MnistDcGan(object):
    def __init__(self):
        self.dcgan = DcGan()

        self.x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
        self.x_train = self.x_train.reshape(-1, self.dcgan.img_size).astype(np.float32)

    def train(self, epoch_size=200, batch_size=128, save_interval=1):
        real_labels = np.ones([batch_size, 1])
        fake_labels = np.zeros([batch_size, 1])
        comp_labels = np.concatenate((real_labels, fake_labels))

        for i in range(epoch_size):
            real_imgs = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :]
            
            fake_imgs = np.random.uniform(-1, 1, size=[batch_size, self.dcgan.latent_size])
            fake_imgs = self.dcgan.G.predict(fake_imgs)
            
            comp_imgs = np.concatenate((real_imgs, fake_imgs))

            # ============================================================ #
            #                    Train the discriminator                   #
            # ============================================================ #
            d_loss = self.dcgan.DM.train_on_batch(comp_imgs, comp_labels)

            # ============================================================ #
            #                    Train the generator                       #
            # ============================================================ #
            fake_imgs = np.random.uniform(-1, 1, size=[batch_size, self.dcgan.latent_size])
            a_loss = self.dcgan.AM.train_on_batch(fake_imgs, real_labels)

            msg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            msg = "%s  [A loss: %f, acc: %f]" % (msg, a_loss[0], a_loss[1])
            print(msg)
            if (i+1) % save_interval == 0:
                fake_img = np.random.uniform(-1, 1, size=[16, self.dcgan.latent_size])
                self.save_image(self.dcgan.G.predict(fake_img), "%s/gimg_%04d.png"%(IMGDIR, i))

    def save_image(self, imgs, path):
        self.enforce_path(path)
        plt.figure(figsize=(10, 10))
        for i in range(imgs.shape[0]):
            plt.subplot(4, 4, i+1)
            img = imgs[i].reshape(self.dcgan.img_row, self.dcgan.img_col)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(path)
        plt.close('all')
    
    def enforce_path(self, path):
        d = os.path.dirname(path)
        if not os.path.exists(d):
            os.mkdir(d)

if __name__ == '__main__':
    mdg = MnistDcGan()
    timer = ElapsedTimer()
    mdg.train(epoch_size=128, batch_size=128, save_interval=1)
    timer.elapsed_time()    