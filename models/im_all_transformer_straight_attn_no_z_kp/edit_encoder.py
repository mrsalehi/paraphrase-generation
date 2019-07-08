import tensorflow as tf

from models.common import graph_utils, vocab
from models.common.config import Config
from models.im_all_transformer import edit_encoder
from models.im_all_transformer.edit_encoder import TransformerMicroEditExtractor
from models.im_all_transformer.transformer import model_utils
from models.im_all_transformer.transformer.embedding_layer import EmbeddingSharedWeights

OPS_NAME = 'edit_encoder'


class EditEncoderNoZ(edit_encoder.EditEncoder):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.config = config

        self.embedding_layer = EmbeddingSharedWeights.get_from_graph()
        if config.editor.word_dim != config.editor.edit_encoder.extractor.hidden_size:
            self.embedding_layer = self.embedding_layer.get_projected(
                config.editor.edit_encoder.extractor.hidden_size)

        self.micro_ev_projection = tf.layers.Dense(
            config.editor.edit_encoder.micro_ev_dim,
            activation=config.editor.edit_encoder.get('mev_proj_activation_fn', None),
            use_bias=True,
            name='micro_ev_proj'
        )

        extractor_config = Config.merge_to_new([config.editor.transformer, config.editor.edit_encoder.extractor])
        extractor_config.put('save_attentions', config.get('eval.save_attentions', False))
        self.mev_extractor = TransformerMicroEditExtractor(
            self.embedding_layer,
            self.micro_ev_projection,
            extractor_config
        )

    # noinspection PyMethodOverriding
    def call(self, src_word_ids, tgt_word_ids,
             insert_word_ids, common_word_ids,
             src_len, tgt_len, iw_len, cw_len, **kwargs):
        with tf.variable_scope('edit_encoder'):
            outputs = self.mev_extractor(src_word_ids, tgt_word_ids, src_len, tgt_len)
            cnx_tgt, tgt_attn_bias, pooled_src, micro_evs_st = outputs

            src_padding = model_utils.get_padding_by_seq_len(src_len)
            src_attn_bias = model_utils.get_padding_bias(None, src_padding)

            src_word_embeds = self.embedding_layer(src_word_ids)

            return tf.constant([[0.]]), (micro_evs_st, src_word_embeds, src_attn_bias), (
                tf.constant([[0.]]), tf.constant([[0.]]), tf.constant([[0.]]))
