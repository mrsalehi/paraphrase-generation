# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of embedding layer with shared weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order

# from official.utils.accelerator import tpu as tpu_utils
from models.common import vocab


class ProjectedEmbedding(tf.layers.Layer):
    def __init__(self, hidden_dim, embedding_layer, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer: EmbeddingSharedWeights = embedding_layer
        self.embedding_proj = tf.layers.Dense(hidden_dim,
                                              activation=None,
                                              use_bias=False,
                                              name='embedding_proj')

    def call(self, inputs, **kwargs):
        embeddings = self.embedding_layer(inputs)
        projected = self.embedding_proj(embeddings)

        return projected

    @staticmethod
    def get_from_graph(hidden_size):
        return EmbeddingSharedWeights.get_from_graph().get_projected(hidden_size)


class EmbeddingSharedWeights(tf.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    def __init__(self, embedding_matrix, method="gather"):
        """Specify characteristic parameters of embedding layer.

    Args:
      vocab_size: Number of tokens in the embedding. (Typically ~32,000)
      hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)
      method: Strategy for performing embedding lookup. "gather" uses tf.gather
        which performs well on CPUs and GPUs, but very poorly on TPUs. "matmul"
        one-hot encodes the indicies and formulates the embedding as a sparse
        matrix multiplication. The matmul formulation is wasteful as it does
        extra work, however matrix multiplication is very fast on TPUs which
        makes "matmul" considerably faster than "gather" on TPUs.
    """
        super(EmbeddingSharedWeights, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.vocab_size = embedding_matrix.shape[0].value
        self.hidden_size = embedding_matrix.shape[-1].value
        self.word_dim = self.hidden_size
        self.shared_weights = embedding_matrix
        if method not in ("gather", "matmul"):
            raise ValueError("method {} must be 'gather' or 'matmul'".format(method))
        self.method = method

    def call(self, x):
        """Get token embeddings of x.

    Args:
      x: An int64 tensor with shape [batch_size, length]
    Returns:
      embeddings: float32 tensor with shape [batch_size, length, embedding_size]
      padding: float32 tensor with shape [batch_size, length] indicating the
        locations of the padding tokens in x.
    """
        with tf.name_scope("embedding"):
            # Create binary mask of size [batch_size, length]
            mask = self.mask(x)

            if self.method == "gather":
                embeddings = tf.gather(self.shared_weights, x)
                embeddings *= tf.expand_dims(mask, -1)

            # Scale embedding by the sqrt of the hidden size
            embeddings *= self.hidden_size ** 0.5

            return embeddings

    def mask(self, x):
        mask = tf.to_float(tf.not_equal(x, 0))
        return mask

    def get_projected(self, hidden_size):
        return ProjectedEmbedding(hidden_size, self)

    def linear(self, x):
        """Computes logits by running x through a linear layer.

    Args:
      x: A float32 tensor with shape [batch_size, length, hidden_size]
    Returns:
      float32 tensor with shape [batch_size, length, vocab_size].
    """
        with tf.name_scope("presoftmax_linear"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            x = tf.reshape(x, [-1, self.hidden_size])
            logits = tf.matmul(x, self.shared_weights, transpose_b=True)

            return tf.reshape(logits, [batch_size, length, self.vocab_size])

    @staticmethod
    def init_from_embedding_matrix():
        embedding_matrix = vocab.get_embeddings()
        embed_layer = EmbeddingSharedWeights(embedding_matrix)
        tf.add_to_collection('embed_layer', embed_layer)

    @staticmethod
    def get_from_graph():
        layer = tf.get_collection('embed_layer')[0]
        return layer
