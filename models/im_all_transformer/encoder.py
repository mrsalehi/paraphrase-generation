import tensorflow as tf

from models.common import graph_utils
from models.im_all_transformer.transformer import model_utils
from models.im_all_transformer.transformer.embedding_layer import EmbeddingSharedWeights
from models.im_all_transformer.transformer.transformer import EncoderStack

OPS_NAME = 'base_sentence_encoder'


class TransformerEncoder(tf.layers.Layer):
    def __init__(self, config, embedding_layer=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        if embedding_layer is None:
            embedding_layer = EmbeddingSharedWeights.get_from_graph()

        if embedding_layer.word_dim != config.hidden_size:
            self.embedding_layer = embedding_layer.get_projected(config.hidden_size)
        else:
            self.embedding_layer = embedding_layer

        self.encoder = EncoderStack(config.to_json(), graph_utils.is_training())

    def _prepare_inputs(self, seq):
        embedded_inputs = self.embedding_layer(seq)
        length = tf.shape(embedded_inputs)[1]

        with tf.name_scope("pos_encoding"):
            pos_encoding = model_utils.get_position_encoding(length, self.config.hidden_size)
            embedded_inputs += pos_encoding

        if self.config.enable_dropout and self.config.layer_postprocess_dropout > 0.:
            embedded_inputs = tf.nn.dropout(embedded_inputs, 1. - self.config.layer_postprocess_dropout)

        return embedded_inputs

    def call(self, seq, seq_len, **kwargs):
        inputs = self._prepare_inputs(seq)

        padding = model_utils.get_padding_by_seq_len(seq_len)
        attention_bias = model_utils.get_padding_bias(None, padding=padding)

        encoded = self.encoder(inputs, attention_bias, padding)

        return encoded, attention_bias
