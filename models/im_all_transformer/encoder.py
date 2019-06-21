import tensorflow as tf

from models.common import graph_utils
from models.common.config import Config
from models.im_all_transformer.transformer import model_utils
from models.im_all_transformer.transformer.embedding_layer import EmbeddingSharedWeights
from models.im_all_transformer.transformer.transformer import EncoderStack

OPS_NAME = 'base_sentence_encoder'


# def transformer_encoder(seq, seq_len, config, name='transformer_encoder'):
#     with tf.variable_scope(name):
#         if config.enable_dropout and config.layer_postprocess_dropout > 0.0:
#             seq = tf.nn.dropout(seq, 1.0 - config.layer_postprocess_dropout)
#
#         padding = model_utils.get_padding_by_seq_len(seq_len)
#         attention_bias = model_utils.get_padding_bias(None, padding=padding)
#
#         encoder = EncoderStack(config.to_json(), graph_utils.is_training())
#         encoded = encoder(seq, attention_bias, padding)
#
#         return encoded, attention_bias
#
#
# def prepare_transformer_input(seq, seq_len, model_hidden_size, embedding_layer=None):
#     if embedding_layer is None:
#         embedding_layer = EmbeddingSharedWeights.get_from_graph()
#
#     word_dim = embedding_layer.hidden_size
#     if word_dim != model_hidden_size:
#         embedding_layer = ProjectedEmbedding(model_hidden_size, embedding_layer)
#
#     embedded_inputs = embedding_layer(seq)
#     with tf.name_scope("pos_encoding"):
#         length = tf.shape(embedded_inputs)[1]
#         pos_encoding = model_utils.get_position_encoding(length, word_dim)
#         encoder_inputs = embedded_inputs + pos_encoding
#
#     return encoder_inputs
#
#
# def base_sent_encoder(base_word_ids, base_len, config):
#     with tf.name_scope(OPS_NAME):
#         encoder_config = Config.merge_to_new([config.editor.transformer, config.editor.base_sent_encoder])
#         base_embeds = prepare_transformer_input(base_word_ids, base_len, encoder_config.hidden_size)
#
#         if encoder_config.enable_dropout and encoder_config.layer_postprocess_dropout > 0.:
#             base_embeds = tf.nn.dropout(base_embeds, 1. - encoder_config.layer_postprocess_dropout)
#
#         output = transformer_encoder(base_embeds, base_len, encoder_config, name='transformer')
#
#         return output


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