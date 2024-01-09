# ------------------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------
import os
import glob
import random
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    def call(self, inputs):
        mean, logvar = inputs
        batch = tf.shape(mean)[0]
        hid = tf.shape(mean)[1]
        eps = tf.random.normal(shape=(batch,hid))
        return  mean + eps*tf.exp(0.5*logvar)


def Encoder(latent_dim):
    input_shape = (24, 240, 240, 3)
    encoder_inputs = keras.Input(shape=input_shape, name='Input_layer')
    x = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', strides=2, kernel_initializer='he_uniform', padding='same', name='Conv3D_1')(encoder_inputs)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), name='MaxPooling3D_1')(x)
    x = layers.Dropout(0.5, name='Dropout_1')(x)
    x = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', strides=2, kernel_initializer='he_uniform', padding='same', name='Conv3D_2')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), name='MaxPooling3D_2')(x)
    x = layers.Dropout(0.5, name='Dropout_2')(x)
    x = layers.Conv3D(16, kernel_size=(3, 3, 3), activation='relu', strides=2, kernel_initializer='he_uniform', padding='same', name='Conv3D_3')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(latent_dim*2, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')


def Decoder(latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(latent_dim*2, activation='relu', name='Dense_3')(latent_inputs)
    x = layers.Dense(3 * 30 * 30 * 1, activation='relu', name='Dense_4')(x)
    x = layers.Reshape((3, 30, 30, 1))(x)
    x = layers.Conv3DTranspose(3, kernel_size=(3, 3, 3), activation='relu', strides=2, kernel_initializer='he_uniform', padding='same', name='ConvTranspose_1')(x)
    x = layers.Conv3DTranspose(3, kernel_size=(3, 3, 3), activation='relu', strides=2, kernel_initializer='he_uniform', padding='same', name='ConvTranspose_2')(x)
    x = layers.Conv3DTranspose(3, kernel_size=(3, 3, 3), activation='sigmoid', strides=2, kernel_initializer='he_uniform', padding='same', name='ConvTranspose_3')(x)
    return keras.Model(latent_inputs, x, name='decoder')


# VAE
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstructon_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            data1, data2 = tf.split(data, 2, 1)
            data1, data2 = tf.one_hot(data1, 3), tf.one_hot(data2, 3)
            z_mean, z_log_var, z = self.encoder(data1)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data2, reconstruction),)
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def fill_tumor_area(img_array, kernel):
    mask = np.zeros_like(img_array)
    mask[img_array==1] = 1
    kernel = np.ones((kernel,kernel), np.uint8)
    for i in range(24):
        try:
            countours, _ = cv2.findContours(mask[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_img = cv2.drawContours(mask[i], countours, -1, color=1, thickness=kernel)
            countours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_img = cv2.drawContours(mask_img, countours,-1, color=1, thickness=-1)
            erosion_img = cv2.erode(mask_img, kernel, iterations=-1)
            img_array[i][erosion_img==1] = 1
        except: pass
    return img_array


def lesion_feature_extractions(input_arrays, batch_size, latent_dim, kernel):
    model_path = os.path.join(os.path.dirname(__file__), 'pretrained_weights', 'weights_3class_re_%s'%str(latent_dim))
    input_arrays[input_arrays==4] = 0
    input_arrays[input_arrays==1] = 2
    input_arrays[input_arrays==3] = 1
    input_arrays = fill_tumor_area(input_arrays, kernel)
    datasets = tf.data.Dataset.from_tensor_slices(tf.one_hot(input_arrays, 3)).batch(batch_size)
    print('Loading VAE model: extracting features of lesions')
    vae = VAE(Encoder(latent_dim), Decoder(latent_dim))
    vae.load_weights(model_path)
    vae.compile(optimizer=keras.optimizers.Adam())
    mean, _, z = vae.encoder.predict(datasets, verbose=1)
    recon_img = vae.decoder.predict(z, verbose=0)
    mean = np.array(mean)
    recon_img = np.argmax(np.array(recon_img), axis=-1).astype(np.uint8)
    return mean, recon_img
