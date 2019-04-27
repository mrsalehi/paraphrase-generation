import tensorflow as tf

import models.common.sequence as sequence
from models.im_attn_ee.edit_encoder import context_encoder, MultiHeadAttention
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

    micro_edit_feed_st = tf.concat([cnx_src, attn_src_tgt], axis=2)  # bs x src_seq_len x 2*word_dim
    micro_edit_feed_ts = tf.concat([cnx_tgt, attn_tgt_src], axis=2)  # bs x src_seq_len x 2*word_dim

    micro_ev_st_creator = tf.layers.Dense(micro_ev_dim, name='micro_ev_st_creator', use_bias=False)
    micro_ev_ts_creator = tf.layers.Dense(micro_ev_dim, name='micro_ev_ts_creator', use_bias=False)
    micro_evs_st = micro_ev_st_creator(micro_edit_feed_st)  # bs x src_seq_len x micro_edit_vec_dim
    micro_evs_ts = micro_ev_ts_creator(micro_edit_feed_ts)  # bs x src_seq_len x micro_edit_vec_dim

    is_training = tf.get_collection('is_training')[0]

    micro_evs_st = tf.layers.batch_normalization(micro_evs_st, training=is_training, name="normalize_st")
    micro_evs_ts = tf.layers.batch_normalization(micro_evs_ts, training=is_training, name="normalize_ts")

    if use_dropout and dropout_keep < 1.:
        micro_evs_st = tf.nn.dropout(micro_evs_st, dropout_keep)
        micro_evs_ts = tf.nn.dropout(micro_evs_ts, dropout_keep)

    return micro_evs_st, micro_evs_ts


def attn_encoder(source_words, target_words, insert_words, delete_words,
                 source_lengths, target_lengths, iw_lengths, dw_lengths,
                 ctx_hidden_dim, ctx_hidden_layer, wa_hidden_dim, wa_hidden_layer, meve_hidden_dim, meve_hidden_layers,
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

        wa = tf.make_template('wa', context_encoder,
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

        return edit_vector, tf.constant([[0]]), tf.constant([[0]])
