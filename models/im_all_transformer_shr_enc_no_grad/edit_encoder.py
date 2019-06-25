import tensorflow as tf

import models.im_all_transformer_shr_enc.edit_encoder as base_edit_encoder

OPS_NAME = 'edit_encoder'


class TransformerMicroEditExtractor(base_edit_encoder.TransformerMicroEditExtractor):
    def _encode_tgt(self, embedded_tgt, tgt_padding, tgt_attention_bias):
        with tf.name_scope('encode_tgt'):
            if self.params.enable_dropout and self.params.layer_postprocess_dropout > 0.0:
                embedded_tgt = tf.nn.dropout(embedded_tgt, 1.0 - self.params.layer_postprocess_dropout)

            encoded = self.target_encoder(embedded_tgt, tgt_attention_bias, tgt_padding)
            encoded = tf.stop_gradient(encoded)

            return encoded
