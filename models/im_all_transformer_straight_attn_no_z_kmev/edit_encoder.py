import tensorflow as tf

from models.im_all_transformer.transformer import model_utils
from models.im_all_transformer_straight_attn_no_z_kp import EditEncoderNoZ

OPS_NAME = 'edit_encoder'


class EditEncoderNoZKmev(EditEncoderNoZ):

    # noinspection PyMethodOverriding
    def call(self, src_word_ids, tgt_word_ids,
             insert_word_ids, common_word_ids,
             src_len, tgt_len, iw_len, cw_len, **kwargs):
        with tf.variable_scope('edit_encoder'):
            outputs = self.mev_extractor(src_word_ids, tgt_word_ids, src_len, tgt_len)
            cnx_tgt, tgt_attn_bias, pooled_src, micro_evs_st = outputs

            src_padding = model_utils.get_padding_by_seq_len(src_len)
            src_attn_bias = model_utils.get_padding_bias(None, src_padding)

            return tf.constant([[0.]]), (micro_evs_st, micro_evs_st, src_attn_bias), (
                tf.constant([[0.]]), tf.constant([[0.]]), tf.constant([[0.]]))
