import tensorflow as tf

from models.common import graph_utils, vocab
from models.im_all_transformer import edit_encoder
from models.im_all_transformer.transformer import model_utils

OPS_NAME = 'edit_encoder'


class TransformerMicroEditExtractorWithRmTok(edit_encoder.TransformerMicroEditExtractor):
    def __init__(self, embedding_layer, mev_projection, params, **kwargs):
        super().__init__(embedding_layer, mev_projection, params, **kwargs)

        self.rm_tok_embedding = self.add_weight('rm_tok_embedding',
                                                (self.params.hidden_size,),
                                                dtype=tf.float32,
                                                trainable=True)

    # noinspection PyMethodOverriding
    def call(self, src, tgt, src_len, tgt_len, **kwargs):
        # Add [REMOVE] token to the beginning of source sequence
        # [batch, length, hidden_size]
        embedded_tgt = self.embedding_layer(tgt)

        # [batch, length+1, hidden_size]
        extended_embedded_tgt = self._add_token_to_beginning(embedded_tgt, self.rm_tok_embedding)
        extended_embedded_tgt += model_utils.get_position_encoding(
            tf.shape(extended_embedded_tgt)[1],
            self.params.hidden_size
        )
        extended_tgt_len = tgt_len + 1

        if self.params.get('noiser_ident_prob', 1) < 1:
            extended_tgt_attention_bias, extended_tgt_padding = self._get_attn_bias_with_dropout(
                extended_tgt_len, uniform_low=1)
        else:
            extended_tgt_padding = model_utils.get_padding_by_seq_len(extended_tgt_len)
            extended_tgt_attention_bias = model_utils.get_padding_bias(None, extended_tgt_padding)

        # Add [CLS] token to the beginning of source sequence
        # [batch, length, hidden_size]
        embedded_src = self.embedding_layer(src)

        # [batch, length+1, hidden_size]
        extended_embedded_src = self._add_cls_token(embedded_src)
        extended_embedded_src += model_utils.get_position_encoding(
            tf.shape(extended_embedded_src)[1],
            self.params.hidden_size
        )
        extended_src_len = src_len + 1

        if self.params.get('noiser_ident_prob', 1) < 1:
            extended_src_attention_bias, extended_src_padding = self._get_attn_bias_with_dropout(
                extended_src_len, uniform_low=1)
        else:
            extended_src_padding = model_utils.get_padding_by_seq_len(extended_src_len)
            extended_src_attention_bias = model_utils.get_padding_bias(None, extended_src_padding)

        # Encode Target
        # [batch, length+1, hidden_size]
        encoded_tgt = self._encode_tgt(extended_embedded_tgt, extended_tgt_padding, extended_tgt_attention_bias)

        # Decode source using the encoded target
        # [batch, length+1, hidden_size]
        decoder_output = self._decode_micro_edit_vectors(extended_embedded_src, extended_src_padding,
                                                         extended_src_attention_bias,
                                                         encoded_tgt, extended_tgt_attention_bias)

        if not graph_utils.is_training() and self.params.save_attentions:
            tf.add_to_collection('TransformerMicroEditExtractor_Attentions', [
                self.target_encoder.self_attn_alignment_history,
                self.mev_decoder.self_attn_alignment_history,
                self.mev_decoder.enc_dec_attn_alignment_history,
            ])

        with tf.name_scope("pooler"):
            # We "pool" the model by simply taking the hidden state corresponding
            # to the first token.
            first_token_tensor = tf.squeeze(decoder_output[:, 0:1, :], axis=1)
            pooled = self.pooling_layer(first_token_tensor)

        # with tf.control_dependencies([prints]):
        # [batch, length, hidden_size]
        micro_ev = self.mev_projection(decoder_output[:, 1:, :])

        return encoded_tgt[:, 1:, :], extended_tgt_attention_bias[:, :, :, 1:], pooled, micro_ev

    def _add_token_to_beginning(self, embedded_seq, tok):
        """
        Args:
            embedded_seq(Tensor): shape: [batch, length, hidden_size]

        Returns:
            Tensor with [batch, length+1, hidden_size]
        """
        # [1, hidden_size]
        cls_embed = tf.reshape(tok, [1, -1])

        batch_size = tf.shape(embedded_seq)[0]

        # [batch, hidden_size]
        cls_embed = tf.tile(cls_embed, [batch_size, 1])

        # [batch, 1, hidden_size)
        cls_embed = tf.expand_dims(cls_embed, axis=1)

        return tf.concat([cls_embed, embedded_seq], axis=1)

    def _add_token_to_end(self, embedded_seq, seq_len, tok):
        """
        Args:
            embedded_seq: shape [batch, length, hidden_size]
            seq_len: shape [batch]

        Returns:
            Tensor with shape [batch, length+1, hidden_size]
        """
        batch_size = tf.shape(embedded_seq)[0]

        pad_ids = tf.fill([batch_size, 1], vocab.SPECIAL_TOKENS.index(vocab.PAD_TOKEN))

        # [batch, 1, hidden_size]
        pad_embedding = self.embedding_layer(pad_ids)

        # [batch, length+1, hidden_size]
        seq = tf.concat([embedded_seq, pad_embedding], axis=1)

        update_indices = tf.reshape(tf.range(0, batch_size), [-1, 1])
        update_indices = tf.concat([update_indices, tf.reshape(seq_len, [-1, 1])], axis=1)

        rm_tok = tf.tile(tf.expand_dims(tok, axis=0), [batch_size, 1])
        delta = tf.scatter_nd(update_indices, rm_tok, tf.shape(seq))

        seq = seq + delta

        return seq
