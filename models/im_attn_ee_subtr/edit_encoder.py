import tensorflow as tf

import models.common.sequence as sequence
from models.im_attn_ee.edit_encoder import MultiHeadAttention, context_encoder, word_aggregator, masked_fill
from models.neural_editor.edit_encoder import sample_vMF

OPS_NAME = 'edit_encoder'


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

    micro_evs_st = attn_src_tgt - cnx_src  # bs x src_seq_len x word_dim
    micro_evs_ts = attn_tgt_src - cnx_tgt  # bs x src_seq_len x word_dim

    if use_dropout and dropout_keep < 1.:
        micro_evs_st = tf.nn.dropout(micro_evs_st, dropout_keep)
        micro_evs_ts = tf.nn.dropout(micro_evs_ts, dropout_keep)

    return micro_evs_st, micro_evs_ts



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

        features = tf.concat([
            max_mev_st,
            max_mev_ts,
            wa_inserted_last,
            wa_deleted_last
        ], axis=1)

        edit_vector = tf.layers.dense(features, edit_dim, use_bias=False, name='encoder_ev')

        if enable_vae:
            edit_vector = sample_vMF(edit_vector, noise_scaler, norm_eps, norm_max)

        return edit_vector, tf.constant([[0]]), tf.constant([[0]])
