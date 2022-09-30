import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
from glob import glob
import numpy as np

gpu_available = tf.config.list_physical_devices('GPU')
print(gpu_available)

section    = 'vae'
run_id     = '0001'
data_name  = 'faces'
RUN_FOLDER = '../run/{}/'.format(section)
RUN_FOLDER += '_'.join([run_id, data_name])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

mode =  'build' #'load' #

# run params
# DATA_FOLDER   = '/home/mariano/Data/celebA/imgs_align/img_align_celeba_png'
DATA_FOLDER   = '/home/est_posgrado_manuel.suarez/data/img_align_celeba'
INPUT_DIM     = (128,128,3)
LATENT_DIM    = 150
BATCH_SIZE    = 384
R_LOSS_FACTOR = 100000  # 10000
EPOCHS        = 400
INITIAL_EPOCH = 0

filenames  = np.array(glob(os.path.join(DATA_FOLDER, '*.jpg')))
n_images        = filenames.shape[0]
steps_per_epoch = n_images//BATCH_SIZE

print('num image files : ', n_images)
print('steps per epoch : ', steps_per_epoch )

AUTOTUNE = tf.data.AUTOTUNE
dataset=tf.keras.utils.image_dataset_from_directory(directory  = DATA_FOLDER,
                                                    labels     = None,
                                                    batch_size = BATCH_SIZE,
                                                    image_size = INPUT_DIM[:2],
                                                    shuffle    = True,).repeat()

dataset = dataset.prefetch(buffer_size=AUTOTUNE)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
dataset = dataset.map(lambda x: (normalization_layer(x)), num_parallel_calls=AUTOTUNE)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6), tight_layout=True)

for images in dataset.take(1):
    for i in range(18):
        ax = plt.subplot(3, 6, i + 1)
        plt.imshow(images[i].numpy())
        plt.axis('off')

plt.savefig("figure_1.png")

print(images.shape)

from tensorflow import keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Flatten, Reshape, Dropout, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential


class Encoder(keras.Model):
    def __init__(self, input_dim, output_dim, encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
                 use_batch_norm=True, use_dropout=True, **kwargs):
        '''
        '''
        super(Encoder, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.n_layers_encoder = len(self.encoder_conv_filters)
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.model = self.encoder_model()
        self.built = True

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({"units": self.units})
        return config

    def encoder_model(self):
        '''
        '''
        encoder_input = layers.Input(shape=self.input_dim, name='encoder')
        x = encoder_input

        for i in range(self.n_layers_encoder):
            x = Conv2D(filters=self.encoder_conv_filters[i],
                       kernel_size=self.encoder_conv_kernel_size[i],
                       strides=self.encoder_conv_strides[i],
                       padding='same',
                       name='encoder_conv_' + str(i), )(x)
            if self.use_batch_norm:
                x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            if self.use_dropout:
                x = Dropout(rate=0.25)(x)

        self.last_conv_size = x.shape[1:]
        x = Flatten()(x)
        encoder_output = Dense(self.output_dim)(x)
        model = keras.Model(encoder_input, encoder_output)
        return model

    def call(self, inputs):
        '''
        '''
        return self.model(inputs)


class Decoder(keras.Model):
    def __init__(self, input_dim, input_conv_dim,
                 decoder_conv_t_filters, decoder_conv_t_kernel_size, decoder_conv_t_strides,
                 use_batch_norm=True, use_dropout=True, **kwargs):

        '''
        '''
        super(Decoder, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.input_conv_dim = input_conv_dim

        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.n_layers_decoder = len(self.decoder_conv_t_filters)

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.model = self.decoder_model()
        self.built = True

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({"units": self.units})
        return config

    def decoder_model(self):
        '''
        '''
        decoder_input = layers.Input(shape=self.input_dim, name='decoder')
        x = Dense(np.prod(self.input_conv_dim))(decoder_input)
        x = Reshape(self.input_conv_dim)(x)

        for i in range(self.n_layers_decoder):
            x = Conv2DTranspose(filters=self.decoder_conv_t_filters[i],
                                kernel_size=self.decoder_conv_t_kernel_size[i],
                                strides=self.decoder_conv_t_strides[i],
                                padding='same',
                                name='decoder_conv_t_' + str(i))(x)
            if i < self.n_layers_decoder - 1:

                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                x = LeakyReLU()(x)
                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:
                x = Activation('sigmoid')(x)
        decoder_output = x
        model = keras.Model(decoder_input, decoder_output)
        return model

    def call(self, inputs):
        '''
        '''
        return self.model(inputs)


class Sampler(keras.Model):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, latent_dim, **kwargs):
        super(Sampler, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.model = self.sampler_model()
        self.built = True

    def get_config(self):
        config = super(Sampler, self).get_config()
        config.update({"units": self.units})
        return config

    def sampler_model(self):
        '''
        input_dim is a vector in the latent (codified) space
        '''
        input_data = layers.Input(shape=self.latent_dim)
        z_mean = Dense(self.latent_dim, name="z_mean")(input_data)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(input_data)

        self.batch = tf.shape(z_mean)[0]
        self.dim = tf.shape(z_mean)[1]

        epsilon = tf.keras.backend.random_normal(shape=(self.batch, self.dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        model = keras.Model(input_data, [z, z_mean, z_log_var])
        return model

    def call(self, inputs):
        '''
        '''
        return self.model(inputs)


class VAE(keras.Model):
    def __init__(self, r_loss_factor=1, summary=False, **kwargs):
        super(VAE, self).__init__(**kwargs)

        self.r_loss_factor = r_loss_factor

        # Architecture
        self.input_dim = INPUT_DIM
        self.latent_dim = LATENT_DIM
        self.encoder_conv_filters = [64, 64, 64, 64]
        self.encoder_conv_kernel_size = [3, 3, 3, 3]
        self.encoder_conv_strides = [2, 2, 2, 2]
        self.n_layers_encoder = len(self.encoder_conv_filters)

        self.decoder_conv_t_filters = [64, 64, 64, 3]
        self.decoder_conv_t_kernel_size = [3, 3, 3, 3]
        self.decoder_conv_t_strides = [2, 2, 2, 2]
        self.n_layers_decoder = len(self.decoder_conv_t_filters)

        self.use_batch_norm = True
        self.use_dropout = True

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.mae = tf.keras.losses.MeanAbsoluteError()

        # Encoder
        self.encoder_model = Encoder(input_dim=self.input_dim,
                                     output_dim=self.latent_dim,
                                     encoder_conv_filters=self.encoder_conv_filters,
                                     encoder_conv_kernel_size=self.encoder_conv_kernel_size,
                                     encoder_conv_strides=self.encoder_conv_strides,
                                     use_batch_norm=self.use_batch_norm,
                                     use_dropout=self.use_dropout)
        self.encoder_conv_size = self.encoder_model.last_conv_size
        if summary:
            self.encoder_model.summary()

        # Sampler
        self.sampler_model = Sampler(latent_dim=self.latent_dim)
        if summary:
            self.sampler_model.summary()

        # Decoder
        self.decoder_model = Decoder(input_dim=self.latent_dim,
                                     input_conv_dim=self.encoder_conv_size,
                                     decoder_conv_t_filters=self.decoder_conv_t_filters,
                                     decoder_conv_t_kernel_size=self.decoder_conv_t_kernel_size,
                                     decoder_conv_t_strides=self.decoder_conv_t_strides,
                                     use_batch_norm=self.use_batch_norm,
                                     use_dropout=self.use_dropout)
        if summary: self.decoder_model.summary()

        self.built = True

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker, ]

    @tf.function
    def train_step(self, data):
        '''
        '''
        with tf.GradientTape() as tape:
            # predict
            x = self.encoder_model(data)
            z, z_mean, z_log_var = self.sampler_model(x)
            pred = self.decoder_model(z)

            # loss
            r_loss = self.r_loss_factor * self.mae(data, pred)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = r_loss + kl_loss

        # gradient
        grads = tape.gradient(total_loss, self.trainable_weights)
        # train step
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # compute progress
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(r_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {"loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(), }

    @tf.function
    def generate(self, z_sample):
        '''
        We use the sample of the N(0,I) directly as
        input of the deterministic generator.
        '''
        return self.decoder_model(z_sample)

    @tf.function
    def codify(self, images):
        '''
        For an input image we obtain its particular distribution:
        its mean, its variance (unvertaintly) and a sample z of such distribution.
        '''
        x = self.encoder_model.predict(images)
        z, z_mean, z_log_var = self.sampler_model(x)
        return z, z_mean, z_log_var

    # implement the call method
    @tf.function
    def call(self, inputs, training=False):
        '''
        '''
        tmp1, tmp2 = self.encoder_model.use_Dropout, self.decoder_model.use_Dropout
        if not training:
            self.encoder_model.use_Dropout, self.decoder_model.use_Dropout = False, False

        x = self.encoder_model(inputs)
        z, z_mean, z_log_var = self.sampler_model(x)
        pred = self.decoder_model(z)

        self.encoder_model.use_Dropout, self.decoder_model.use_Dropout = tmp1, tmp2
        return pred

vae = VAE(r_loss_factor=R_LOSS_FACTOR, summary=True)
vae.summary()