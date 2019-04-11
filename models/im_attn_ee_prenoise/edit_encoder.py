import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as tf_rnn
from tensorflow.keras.layers import Layer

import models.common.sequence as sequence
from models.neural_editor.edit_encoder import sample_vMF

OPS_NAME = 'edit_encoder'


def word_aggregator(words, lengths, hidden_dim, num_layers, swap_memory=False, use_dropout=False, dropout_keep=1.0,
                    name=None, reuse=None):
    """
    Args:
        words: tensor in shape of [batch x max_len x embed_dim]
        lengths: tensor in shape of [batch]
        hidden_dim: num of lstm hidden units
        name: op name
        reuse: reuse variable

    Returns:
        aggregation_result, a tensor in shape of [batch x max_len x hidden_dim]

    """
    with tf.variable_scope(name, 'word_aggregator', [words, lengths], reuse=reuse):
        batch_size = tf.shape(words)[0]

        def create_rnn_layer(layer_num, dim):
            cell = tf_rnn.LSTMCell(dim, name='layer_%s' % layer_num)
            if use_dropout and dropout_keep < 1.:
                cell = tf_rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep)

            if layer_num > 0:
                cell = tf_rnn.ResidualWrapper(cell)

            return cell

        cell = tf_rnn.MultiRNNCell([create_rnn_layer(i, hidden_dim) for i in range(num_layers)])
        zero_state = sequence.create_trainable_initial_states(batch_size, cell)

        outputs, last_state = tf.nn.dynamic_rnn(
            cell,
            words,
            sequence_length=lengths,
            initial_state=zero_state,
            swap_memory=swap_memory
        )

        return outputs


def context_encoder(words, lengths, hidden_dim, num_layers, swap_memory=False, use_dropout=False, dropout_keep=1.0,
                    name=None, reuse=None):
    """
    Args:
        words: tensor in shape of [batch x max_len x embed_dim]
        lengths: tensor in shape of [batch]
        hidden_dim: num of lstm hidden units
        name: op name
        reuse: reuse variable

    Returns:
        aggregation_result, a tensor in shape of [batch x max_len x hidden_dim]

    """
    with tf.variable_scope(name, 'context_encoder', [words, lengths], reuse=reuse):
        def create_rnn_layer(layer_num, dim):
            cell = tf_rnn.LSTMCell(dim, name='layer_%s' % layer_num)
            if use_dropout and dropout_keep < 1.:
                cell = tf_rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep)

            if layer_num > 0:
                cell = tf_rnn.ResidualWrapper(cell)

            return cell

        batch_size = tf.shape(words)[0]

        fw_cell = tf_rnn.MultiRNNCell([create_rnn_layer(i, hidden_dim // 2) for i in range(num_layers)])
        bw_cell = tf_rnn.MultiRNNCell([create_rnn_layer(i, hidden_dim // 2) for i in range(num_layers)])

        fw_zero_state = sequence.create_trainable_initial_states(batch_size, fw_cell, 'fw_zs')
        bw_zero_state = sequence.create_trainable_initial_states(batch_size, bw_cell, 'bw_zs')

        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            words,
            lengths,
            fw_zero_state,
            bw_zero_state,
            swap_memory=swap_memory
        )

        output = tf.concat(outputs, axis=2)
        assert output.shape[2] == hidden_dim

    return output


class AttentionHead(Layer):
    def __init__(self, d_k, d_v, temperature, **kwargs):
        super().__init__(**kwargs)

        self.d_k = d_k
        self.d_v = d_v
        self.temperature = temperature

        self.w_q = tf.layers.Dense(self.d_k, activation=None, use_bias=None)
        self.w_k = tf.layers.Dense(self.d_k, activation=None, use_bias=None)
        self.w_v = tf.layers.Dense(self.d_v, activation=None, use_bias=None)

    def call(self, inputs, **kwargs):
        """

        Args:
            q (Tensor): bs x len_q x word_dim
            k (Tensor): bs x len_k x word_dim
            v (Tensor): bs x len_v x word_dim

            memory_lengths (Tensor): bs
            **kwargs:

        Returns:

        """
        q, k, v, memory_lengths = inputs
        len_q = tf.shape(q)[1]

        qs, ks, vs = self.w_q(q), self.w_k(k), self.w_v(v)  # bs x len x d_small

        attn = tf.matmul(qs, ks, transpose_b=True)  # bs x len_q x len_k
        attn = attn / self.temperature

        mem_mask = tf.sequence_mask(memory_lengths)
        mem_mask = tf.tile(tf.expand_dims(mem_mask, axis=1), [1, len_q, 1])
        padding = tf.ones_like(mem_mask, dtype=tf.float32) * -1e9
        attn = tf.where(tf.equal(mem_mask, True), attn, padding)

        attn = tf.nn.softmax(attn)  # bs x len_q x len_k
        attn_result = tf.matmul(attn, vs)  # bs x len_q x d_small

        return attn_result


class MultiHeadAttention(Layer):
    def __init__(self, num_heads, d_model, d_small, use_dropout=False, dropout_keep=1., **kwargs):
        super().__init__(**kwargs)

        self.use_dropout = use_dropout
        self.use_dropout = dropout_keep

        temp = np.power(d_small, 0.5)
        self.heads = [AttentionHead(d_small, d_small, temp, name='head_%s' % i) for i in range(num_heads)]

        # [num_heads * d_small, d_model]
        self.w_o = tf.layers.Dense(d_model, activation=None, use_bias=None)

    def call(self, inputs, **kwargs):
        # list of [bs, len_q, d_small]
        head_result = []
        for head in self.heads:
            head_result.append(head(inputs))

        attn = tf.concat(head_result, axis=2)
        attn = self.w_o(attn)

        return attn


def create_micro_edit_vectors(cnx_src, cnx_tgt, src_lengths, tgt_lengths,
                              d_model, num_heads, micro_ev_dim,
                              dropout_keep=1., use_dropout=False):
    assert d_model % num_heads == 0
    d_small = int(d_model // num_heads)

    st_mha = MultiHeadAttention(num_heads, d_model, d_small, use_dropout, dropout_keep, name='src_tgt_attn')
    ts_mha = MultiHeadAttention(num_heads, d_model, d_small, use_dropout, dropout_keep, name='tgt_src_attn')

    attn_src_tgt = st_mha([cnx_src, cnx_tgt, cnx_tgt, tgt_lengths])  # bs x src_seq_len x word_dim
    attn_tgt_src = ts_mha([cnx_tgt, cnx_src, cnx_src, src_lengths])  # bs x tgt_seq_len x word_dim

    if use_dropout and dropout_keep < 1.:
        attn_src_tgt = tf.nn.dropout(attn_src_tgt, dropout_keep)
        attn_tgt_src = tf.nn.dropout(attn_tgt_src, dropout_keep)

    micro_edit_feed_st = tf.concat([cnx_src, attn_src_tgt], axis=2)  # bs x src_seq_len x 2*word_dim
    micro_edit_feed_ts = tf.concat([cnx_tgt, attn_tgt_src], axis=2)  # bs x src_seq_len x 2*word_dim

    micro_ev_creator = tf.layers.Dense(micro_ev_dim, name='micro_ev_creator')
    micro_evs_st = micro_ev_creator(micro_edit_feed_st)  # bs x src_seq_len x micro_edit_vec_dim
    micro_evs_ts = micro_ev_creator(micro_edit_feed_ts)  # bs x src_seq_len x micro_edit_vec_dim

    if use_dropout and dropout_keep < 1.:
        micro_evs_st = tf.nn.dropout(micro_evs_st, dropout_keep)
        micro_evs_ts = tf.nn.dropout(micro_evs_ts, dropout_keep)

    return micro_evs_st, micro_evs_ts


def masked_fill(tensor, seq_lengths, mask_value):
    seq_len = tf.shape(tensor)[2]

    mask = tf.sequence_mask(seq_lengths)
    mask = tf.tile(tf.expand_dims(mask, axis=-1), [1, 1, seq_len])
    padding = tf.ones_like(mask, dtype=tf.float32) * mask_value
    masked = tf.where(tf.equal(mask, True), tensor, padding)

    return masked


def attn_encoder(source_words, target_words, insert_words, delete_words,
                 source_lengths, target_lengths, iw_lengths, dw_lengths,
                 ctx_hidden_dim, ctx_hidden_layer, wa_hidden_dim, wa_hidden_layer,
                 edit_dim, micro_edit_ev_dim, num_heads, noise_scaler, norm_eps, norm_max,
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
    print("RUn me111!!")
    with tf.variable_scope(OPS_NAME):
        cnx_encoder = tf.make_template('cnx_encoder', context_encoder,
                                       hidden_dim=ctx_hidden_dim,
                                       num_layers=ctx_hidden_layer,
                                       swap_memory=swap_memory,
                                       use_dropout=use_dropout,
                                       dropout_keep=dropout_keep)

        wa = tf.make_template('wa', word_aggregator,
                              hidden_dim=wa_hidden_dim,
                              num_layers=wa_hidden_layer,
                              swap_memory=swap_memory,
                              use_dropout=use_dropout,
                              dropout_keep=dropout_keep)

        wa_inserted = wa(insert_words, iw_lengths)
        wa_deleted = wa(delete_words, dw_lengths)

        wa_inserted_last = sequence.last_relevant(wa_inserted, iw_lengths)
        wa_deleted_last = sequence.last_relevant(wa_deleted, dw_lengths)

        if use_dropout and dropout_keep < 1.:
            wa_inserted_last = tf.nn.dropout(wa_inserted_last, dropout_keep)
            wa_deleted_last = tf.nn.dropout(wa_deleted_last, dropout_keep)

        cnx_src = cnx_encoder(source_words, source_lengths)
        cnx_tgt = cnx_encoder(target_words, target_lengths)

        # bs x seq_len x micro_edit_vec_dim
        micro_evs_st, micro_evs_ts = create_micro_edit_vectors(
            cnx_src, cnx_tgt, source_lengths, target_lengths,
            ctx_hidden_dim, num_heads, micro_edit_ev_dim,
            dropout_keep, use_dropout
        )

        micro_evs_st = masked_fill(micro_evs_st, source_lengths, -1e9)
        micro_evs_ts = masked_fill(micro_evs_ts, target_lengths, -1e9)

        max_mev_st = tf.reduce_max(micro_evs_st, axis=1)  # bs x micro_edit_vec_dim
        max_mev_ts = tf.reduce_max(micro_evs_ts, axis=1)  # bs x micro_edit_vec_dim

        micro_ev_final_nodes = int(micro_edit_ev_dim / (micro_edit_ev_dim + wa_hidden_dim) * edit_dim)
        wa_final_nodes = int(wa_hidden_dim / (micro_edit_ev_dim + wa_hidden_dim) * edit_dim)

        micro_evs_prenoise = tf.layers.Dense(micro_ev_final_nodes // 2, activation=None, use_bias=False)
        wa_prenoise = tf.layers.Dense(wa_final_nodes // 2, activation=None, use_bias=False)

        edit_vector = tf.concat([
            micro_evs_prenoise(max_mev_st),
            micro_evs_prenoise(max_mev_ts),
            wa_prenoise(wa_inserted_last),
            wa_prenoise(wa_deleted_last)
        ], axis=1)

        if enable_vae:
            edit_vector = sample_vMF(edit_vector, noise_scaler, norm_eps, norm_max)

        return edit_vector, None, None
