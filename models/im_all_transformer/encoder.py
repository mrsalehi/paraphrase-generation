import tensorflow as tf

from models.common import graph_utils
from models.common.config import Config
from models.im_all_transformer.transformer import model_utils
from models.im_all_transformer.transformer.embedding_layer import EmbeddingSharedWeights
from models.im_all_transformer.transformer.transformer import EncoderStack

OPS_NAME = 'base_sentence_encoder'


def transformer_encoder(seq, seq_len, config, name='transformer_encoder'):
    with tf.variable_scope(name):
        if config.enable_dropout and config.layer_postprocess_dropout > 0.0:
            seq = tf.nn.dropout(seq, 1.0 - config.layer_postprocess_dropout)

        padding = model_utils.get_padding_by_seq_len(seq_len)
        attention_bias = model_utils.get_padding_bias(None, padding=padding)

        encoder = EncoderStack(config.to_json(), graph_utils.is_training())
        encoded = encoder(seq, attention_bias, padding)

        return encoded


def prepare_transformer_input(seq, seq_len, embedding_layer=None):
    if embedding_layer is None:
        embedding_layer = EmbeddingSharedWeights.get_from_graph()

    hidden_size = embedding_layer.hidden_size

    embedded_inputs = embedding_layer(seq)
    with tf.name_scope("add_pos_encoding"):
        length = tf.shape(embedded_inputs)[1]
        pos_encoding = model_utils.get_position_encoding(length, hidden_size)
        encoder_inputs = embedded_inputs + pos_encoding

    return encoder_inputs


def base_sent_encoder(base_word_ids, base_len, config):
    with tf.name_scope(OPS_NAME):
        encoder_config = Config.merge([config.editor.transformer, config.editor.base_sent_encoder])
        embedded_base = prepare_transformer_input(base_word_ids, base_len)
        encoded = transformer_encoder(embedded_base, base_len, encoder_config, name='transformer')

        return encoded
