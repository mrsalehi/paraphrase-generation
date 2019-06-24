import tensorflow as tf

import models.common.sequence as sequence
from models.common import vocab, graph_utils
from models.im_attn_ee.edit_encoder import context_encoder
from models.im_transf_ee_rnn.transformer import model_utils
from models.im_transf_ee_rnn.transformer.transformer import EncoderStack, DecoderStack
from models.neural_editor.edit_encoder import sample_vMF

OPS_NAME = 'edit_encoder'


class TransformerMicroEditExtractor(tf.layers.Layer):
    def __init__(self, embedding_layer, mev_projection, params, **kwargs):
        super().__init__(**kwargs)
        self.params = params
        is_training = tf.get_collection('is_training')[0]

        self.target_encoder = EncoderStack(self.params.to_json(), is_training)
        self.mev_decoder = DecoderStack(self.params.to_json(), is_training)

        self.embedding_layer = embedding_layer
        self.mev_projection = mev_projection

    def call(self, src, tgt=None, src_len=None, tgt_len=None, **kwargs):
        assert src is not None \
               and tgt is not None \
               and src_len is not None \
               and tgt_len is not None

        initializer = tf.variance_scaling_initializer(
            self.params.initializer_gain, mode="fan_avg", distribution="uniform")

        with tf.variable_scope("TMEV", initializer=initializer):
            embedded_tgt = self.embedding_layer(tgt, tgt_len)
            tgt_padding = model_utils.get_padding(tgt)
            tgt_attention_bias = model_utils.get_padding_bias(None, tgt_padding)

            embedded_src = self.embedding_layer(src, src_len)
            src_padding = model_utils.get_padding(src)
            src_attention_bias = model_utils.get_padding_bias(None, src_padding)

            encoded_tgt = self._encode_tgt(embedded_tgt, tgt_padding, tgt_attention_bias)

            micro_ev = self._decode_micro_edit_vectors(embedded_src, src_padding, src_attention_bias,
                                                       encoded_tgt, tgt_attention_bias)

            if not graph_utils.is_training():
                tf.add_to_collection('TransformerMicroEditExtractor_Attentions', [
                    self.target_encoder.self_attn_alignment_history,
                    self.mev_decoder.self_attn_alignment_history,
                    self.mev_decoder.enc_dec_attn_alignment_history,
                ])

            return encoded_tgt, micro_ev

    def _encode_tgt(self, embedded_tgt, tgt_padding, tgt_attention_bias):
        with tf.name_scope('encode_tgt'):
            if self.params.enable_dropout and self.params.layer_postprocess_dropout > 0.0:
                embedded_tgt = tf.nn.dropout(embedded_tgt, 1.0 - self.params.layer_postprocess_dropout)

            return self.target_encoder(embedded_tgt, tgt_attention_bias, tgt_padding)

    def _decode_micro_edit_vectors(self, embedded_src, src_padding, src_attention_bias,
                                   encoded_tgt, tgt_attention_bias):
        with tf.name_scope('decode_mev'):
            if self.params.enable_dropout and self.params.layer_postprocess_dropout > 0.0:
                embedded_src = tf.nn.dropout(embedded_src, 1 - self.params.layer_postprocess_dropout)

            outputs = self.mev_decoder(embedded_src, encoded_tgt,
                                       src_attention_bias, tgt_attention_bias,
                                       input_padding=src_padding)

            micro_ev = self.mev_projection(outputs)
            return micro_ev


class ConcatPosEmbedding(tf.layers.Layer):
    def __init__(self, hidden_dim, embeddings_matrix, pos_encoding_dim, **kwargs):
        super().__init__(**kwargs)
        self.embeddings_matrix = embeddings_matrix
        self.pos_encoding_dim = pos_encoding_dim
        self.hidden_dim = hidden_dim
        self.hidden_proj = tf.layers.Dense(hidden_dim, activation=None, use_bias=False, name='hidden_proj')

    def call(self, inputs, inputs_len=None, **kwargs):
        batch_size = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]

        with tf.name_scope('mask'):
            mask = tf.to_float(tf.not_equal(inputs, 0))

        with tf.name_scope('embedding_lookup'):
            embed = tf.nn.embedding_lookup(self.embeddings_matrix, inputs)

        with tf.name_scope('pos_encoding'):
            pos_encoding = model_utils.get_position_encoding(length, self.pos_encoding_dim)
            pos_encoding = tf.tile(tf.expand_dims(pos_encoding, 0), [batch_size, 1, 1])

        pos_embed = tf.concat([pos_encoding, embed], axis=-1)

        output = self.hidden_proj(pos_embed)
        output *= self.hidden_dim ** 0.5
        output *= tf.expand_dims(mask, -1)

        return output


def wa_accumulator(insert_words, delete_words,
                   iw_lengths, dw_lengths,
                   edit_dim):
    max_len = tf.shape(insert_words)[1]
    mask = tf.sequence_mask(iw_lengths, maxlen=max_len, dtype=tf.float32)
    mask = tf.expand_dims(mask, 2)
    insert_embed = tf.reduce_sum(mask * insert_words, axis=1)

    max_len = tf.shape(delete_words)[1]
    mask = tf.sequence_mask(dw_lengths, maxlen=max_len, dtype=tf.float32)
    mask = tf.expand_dims(mask, 2)
    delete_embed = tf.reduce_sum(mask * delete_words, axis=1)

    linear_prenoise = tf.make_template('linear_prenoise', tf.layers.dense,
                                       units=edit_dim,
                                       activation=None,
                                       use_bias=False)
    insert_embed = linear_prenoise(insert_embed)
    delete_embed = linear_prenoise(delete_embed)

    return insert_embed, delete_embed


def attn_encoder(source_words, target_words, insert_words, delete_words,
                 source_lengths, target_lengths, iw_lengths, dw_lengths,
                 transformer_params, wa_hidden_dim, meve_hidden_dim,
                 meve_hidden_layers,
                 edit_dim, micro_edit_ev_dim, noise_scaler, norm_eps, norm_max,
                 dropout_keep=1., use_dropout=False, swap_memory=False, enable_vae=True):
    """
    Args:
        source_words:
        target_words:
        insert_words:
        delete_words:
        source_lengths:
        target_lengths:
        iw_lengths:
        dw_lengths:
        ctx_hidden_dim:
        ctx_hidden_layer:
        wa_hidden_dim:
        wa_hidden_layer:
        edit_dim:
        noise_scaler:
        norm_eps:
        norm_max:
        dropout_keep:

    Returns:

    """
    with tf.variable_scope(OPS_NAME):
        wa_inserted_last, wa_deleted_last = wa_accumulator(insert_words, delete_words,
                                                           iw_lengths, dw_lengths, wa_hidden_dim)

        if use_dropout and dropout_keep < 1.:
            wa_inserted_last = tf.nn.dropout(wa_inserted_last, dropout_keep)
            wa_deleted_last = tf.nn.dropout(wa_deleted_last, dropout_keep)

        embedding_matrix = vocab.get_embeddings()
        embedding_layer = ConcatPosEmbedding(transformer_params.hidden_size, embedding_matrix,
                                             transformer_params.pos_encoding_dim)
        micro_ev_projection = tf.layers.Dense(micro_edit_ev_dim, activation=None,
                                              use_bias=True, name='micro_ev_proj')
        mev_extractor = TransformerMicroEditExtractor(embedding_layer, micro_ev_projection, transformer_params)

        cnx_tgt, micro_evs_st = mev_extractor(source_words, target_words, source_lengths, target_lengths)
        cnx_src, micro_evs_ts = mev_extractor(target_words, source_words, target_lengths, source_lengths)

        micro_ev_encoder = tf.make_template('micro_ev_encoder', context_encoder,
                                            hidden_dim=meve_hidden_dim,
                                            num_layers=meve_hidden_layers,
                                            swap_memory=swap_memory,
                                            use_dropout=use_dropout,
                                            dropout_keep=dropout_keep)

        aggreg_mev_st = micro_ev_encoder(micro_evs_st, source_lengths)
        aggreg_mev_ts = micro_ev_encoder(micro_evs_ts, target_lengths)

        aggreg_mev_st_last = sequence.last_relevant(aggreg_mev_st, source_lengths)
        aggreg_mev_ts_last = sequence.last_relevant(aggreg_mev_ts, target_lengths)

        if use_dropout and dropout_keep < 1.:
            aggreg_mev_st_last = tf.nn.dropout(aggreg_mev_st_last, dropout_keep)
            aggreg_mev_ts_last = tf.nn.dropout(aggreg_mev_ts_last, dropout_keep)

        features = tf.concat([
            aggreg_mev_st_last,
            aggreg_mev_ts_last,
            wa_inserted_last,
            wa_deleted_last
        ], axis=1)

        edit_vector = tf.layers.dense(features, edit_dim, use_bias=False, name='encoder_ev')

        if enable_vae:
            edit_vector = sample_vMF(edit_vector, noise_scaler, norm_eps, norm_max)

        return edit_vector, (cnx_src, micro_evs_st), (cnx_tgt, micro_evs_ts)
