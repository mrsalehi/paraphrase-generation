import tensorflow as tf

from models.common import graph_utils, vocab
from models.im_all_transformer import edit_encoder
from models.im_all_transformer.edit_encoder import TransformerMicroEditExtractor
from models.im_all_transformer.transformer import model_utils
from models.im_all_transformer.transformer.embedding_layer import EmbeddingSharedWeights
from models.common.config import Config


OPS_NAME = 'edit_encoder'

class EditEncoderRemovePMinusQ(edit_encoder.EditEncoder):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

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

        self.edit_vector_projection = tf.layers.Dense(
            config.editor.edit_encoder.edit_dim,
            activation=config.editor.edit_encoder.get('edit_vector_proj_activation_fn', None),
            use_bias=False,
            name='encoder_ev'
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
            # orig_embedding_layer = EmbeddingSharedWeights.get_from_graph()
            # wa_inserted = self.wa(orig_embedding_layer(insert_word_ids), iw_len)
            # wa_common = self.wa(orig_embedding_layer(common_word_ids), iw_len)

            # if self.config.editor.enable_dropout and self.config.editor.dropout > 0.:
            #     wa_inserted = tf.nn.dropout(wa_inserted, 1. - self.config.editor.dropout)
            #     wa_common = tf.nn.dropout(wa_common, 1. - self.config.editor.dropout)

            outputs = self.mev_extractor(src_word_ids, tgt_word_ids, src_len, tgt_len)
            cnx_tgt, tgt_attn_bias, pooled_src, micro_evs_st = outputs

            outputs = self.mev_extractor(tgt_word_ids, src_word_ids, tgt_len, src_len)
            cnx_src, src_attn_bias, pooled_tgt, micro_evs_ts = outputs

            features = tf.concat([
                pooled_src,
                pooled_tgt,
            ], axis=1)

            edit_vector = self.edit_vector_projection(features)

            if self.config.editor.enable_dropout and self.config.editor.dropout > 0.:
                edit_vector = tf.nn.dropout(edit_vector, 1. - self.config.editor.dropout)

            return edit_vector, (micro_evs_st, cnx_src, src_attn_bias), (micro_evs_ts, cnx_tgt, tgt_attn_bias)
