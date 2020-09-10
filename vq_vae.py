'''This file contains 1 codebook for all 4 elements
'''

import gin
import tensorflow as tf
import tensorflow.keras.layers as tfkl
import numpy as np
from ddsp.training.models import Autoencoder


@gin.register
class VectorQuantizer(tfkl.Layer):
    def __init__(self, num_embeddings, 
                 initializer='uniform',      # initialization for z
                 initializer_f0=tf.random_uniform_initializer(minval=0.0,maxval=500.0),     # initialization for f0_hz
                 initializer_f0_ld_scaled=tf.random_uniform_initializer(minval=0.0,maxval=1.0),   # initialization for f0_scaled, ld_scaled
                 with_inds=False, 
                 **kwargs):
        self.num_embeddings = num_embeddings
        self.initializer = initializer
        self.initializer_f0 = initializer_f0
        self.initializer_f0_ld_scaled = initializer_f0_ld_scaled
        self.with_inds = with_inds
        self.embedding_dim = None
        self.embedding_dim_f0 = None
        self.codebook_all = None
        self.codebook = None
        self.codebook_f0 = None
        self.codebook_f0_scaled = None
        self.codebook_ld_scaled = None
        super(VectorQuantizer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Add embedding weights to codebooks.
        # z
        self.embedding_dim = input_shape['z'][-1]
        self.codebook = self.add_weight(name='codebook',
                                        shape=(self.num_embeddings, self.embedding_dim),
                                        initializer=self.initializer,
                                        trainable=True)
        #f0_hz
        self.embedding_dim_f0 = input_shape['f0_hz'][-1]
        self.codebook_f0 = self.add_weight(name='codebook_f0_hz',
                                           shape=(self.num_embeddings, self.embedding_dim_f0),
                                           initializer=self.initializer_f0,
                                           trainable=True)
        #f0_scaled
        self.codebook_f0_scaled = self.add_weight(name='codebook_f0_scaled',
                                           shape=(self.num_embeddings, self.embedding_dim_f0),
                                           initializer=self.initializer_f0_ld_scaled,
                                           trainable=True)
        #ld_scaled
        self.codebook_ld_scaled = self.add_weight(name='codebook_ld_scaled',
                                           shape=(self.num_embeddings, self.embedding_dim_f0),
                                           initializer=self.initializer_f0_ld_scaled,
                                           trainable=True)
        # concate all codesbooks togather
        self.codebook_all = tf.concat([self.codebook_f0, self.codebook_f0_scaled, self.codebook_ld_scaled, self.codebook], axis=-1)
        # Finalize building.
        super(VectorQuantizer, self).build(input_shape)

    def call(self, conditioning, **kwargs):
        # concat f0_hz, f0_scaled,ld_scaled,z togather as encodings
        encodings = tf.concat([conditioning['f0_hz'], conditioning['f0_scaled'], conditioning['ld_scaled'], conditioning['z']], axis=-1)
        latent_preq = encodings

        embed_dim = self.embedding_dim + self.embedding_dim_f0 + self.embedding_dim_f0 + self.embedding_dim_f0
        encodings_flat = tf.reshape(encodings, shape=(-1, embed_dim))
        distances = tf.reduce_mean((encodings_flat[:, None] - self.codebook_all[None]) ** 2, axis=-1)
        min_indices = tf.argmin(distances, axis=-1)
        codes = tf.gather(self.codebook_all, min_indices, axis=0)
        codes = tf.reshape(codes, shape=[-1, *encodings.shape[1:]])
        
        codes_f0, codes_fs, codes_ls, codes_z = tf.split(codes, [1,1,1,16], axis=-1)
        encodings_f0, encodings_fs, encodings_ls, encodings_z = tf.split(encodings, [1,1,1,16], axis=-1)

        # Straight through estimator trick
        # WARNING: codes_forward will not be equal to codes_for_loss, presumably due to numerical instabilities
        codes_for_loss = codes
        codes_forward= encodings_z + tf.stop_gradient(codes_z - encodings_z)
        conditioning['z'] = codes_forward
        codes_forward= encodings_f0 + tf.stop_gradient(codes_f0 - encodings_f0)
        conditioning['f0_hz'] = codes_forward
        codes_forward= encodings_fs + tf.stop_gradient(codes_fs - encodings_fs)
        conditioning['f0_scaled'] = codes_forward
        codes_forward= encodings_ls + tf.stop_gradient(codes_ls - encodings_ls)
        conditioning['f0_scaled'] = codes_forward
        
        conditioning['latent_preq'] = latent_preq
        conditioning['latent_for_loss'] = codes_for_loss
        
        return conditioning, min_indices

@gin.configurable
class QuantizingAutoencoder(Autoencoder):
    def __init__(self, num_embeddings=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = VectorQuantizer(num_embeddings)

    def call(self, features, training=True):
        """Run the core of the network, get predictions and loss."""
        conditioning = self.encode(features, training=training)
        conditioning, _ = self.quantizer(conditioning)
        audio_gen = self.decode(conditioning, training=training)
        if training:
            for loss_obj in self.loss_objs:
                if isinstance(loss_obj, QuantizationLoss):
                    loss = loss_obj(conditioning['latent_for_loss'], conditioning['latent_preq'])
                else:
                    loss = loss_obj(features['audio'], audio_gen)
                self._losses_dict[loss_obj.name] = loss
        return audio_gen


@gin.register
class QuantizationLoss(tfkl.Layer):
    def __init__(self, weight, name):
        super().__init__(name=name)
        self.weight = weight

    # noinspection PyMethodOverriding,PyUnresolvedReferences
    def call(self, codes, encodings):
        return self.calc_loss(codes, encodings) * self.weight

    def calc_loss(self, codes, encodings):
        pass


@gin.register
class CodebookLoss(QuantizationLoss):

    def __init__(self, weight, name='codebook_loss'):
        super().__init__(weight=weight, name=name)

    # noinspection PyMethodOverriding
    def calc_loss(self, codes, encodings):
        """
        Calculates the codebook loss.
        :param codes: The discrete codes that are the result of the quantization process.
        :param encodings: The original non-discrete values that were generated by the encoder.
        :return: The loss.
        """
        return tf.reduce_mean((tf.stop_gradient(encodings) - codes) ** 2)


@gin.register
class CommitmentLoss(QuantizationLoss):

    def __init__(self, weight, name='commitment_loss'):
        super().__init__(weight=weight, name=name)

    # noinspection PyMethodOverriding
    def calc_loss(self, codes, encodings):
        """
        Calculates the commitment loss.
        :param codes: The discrete codes that are the result of the quantization process.
        :param encodings: The original non-discrete values that were generated by the encoder.
        :return: The loss.
        """
        return tf.reduce_mean((encodings - tf.stop_gradient(codes)) ** 2)
