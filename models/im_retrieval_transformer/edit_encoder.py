import tensorflow as tf

from models.common import graph_utils, vocab
from models.common.config import Config
from models.im_all_transformer import edit_encoder
from models.im_all_transformer.edit_encoder import TransformerMicroEditExtractor, WordEmbeddingAccumulator
from models.im_all_transformer.transformer import model_utils
from models.im_all_transformer.transformer.embedding_layer import EmbeddingSharedWeights

OPS_NAME = 'edit_encoder'


class EditEncoderAcc(tf.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.config = config

        config.accumulated_dim = config.editor.edit_encoder.edit_dim // 2
        self.wa = WordEmbeddingAccumulator(config)

    # noinspection PyMethodOverriding
    def call(self, src_word_ids, tgt_word_ids,
             insert_word_ids, common_word_ids,
             src_len, tgt_len, iw_len, cw_len, **kwargs):
        with tf.variable_scope('edit_encoder'):
            orig_embedding_layer = EmbeddingSharedWeights.get_from_graph()

            wa_inserted = self.wa(orig_embedding_layer(insert_word_ids), iw_len)
            wa_common = self.wa(orig_embedding_layer(common_word_ids), iw_len)

            edit_vector = tf.concat([wa_inserted, wa_common], axis=1)

            if self.config.editor.enable_dropout and self.config.editor.dropout > 0.:
                edit_vector = tf.nn.dropout(edit_vector, 1. - self.config.editor.dropout)

            return edit_vector, (tf.constant([[0.0]]), tf.constant([[0.0]]), tf.constant([[0.0]])), \
                   (tf.constant([[0.0]]), tf.constant([[0.0]]), tf.constant([[0.0]]))
