import tensorflow as tf

from models.im_all_transformer import edit_encoder as base_edit_encoder

OPS_NAME = 'edit_encoder'


class EditEncoder(base_edit_encoder.EditEncoder):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        del self.wa

    # noinspection PyMethodOverriding
    def call(self, src_word_ids, tgt_word_ids,
             insert_word_ids, common_word_ids,
             src_len, tgt_len, iw_len, cw_len, **kwargs):
        with tf.variable_scope('edit_encoder'):
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
