import tensorflow as tf

import models.common.sequence as sequence
from models.im_attn_ee.edit_encoder import context_encoder, MultiHeadAttention
from models.neural_editor.edit_encoder import sample_vMF

OPS_NAME = 'edit_encoder'


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


def create_micro_edit_vectors(cnx_src, cnx_tgt, src_lengths, tgt_lengths,
                              d_model, num_heads, micro_ev_dim,
                              dropout_keep=1., use_dropout=False):
    batch_size = tf.shape(cnx_src)[0]
    memory_size = cnx_src.shape[-1]
    remove_embedding = tf.get_variable('remove_tok_embedding',
                                       shape=(memory_size,),
                                       dtype=tf.float32,
                                       initializer=tf.initializers.glorot_uniform(),
                                       trainable=True)

    # 1 x 1 x memory_size
    remove_embedding = tf.expand_dims(tf.expand_dims(remove_embedding, 0), 0)

    # bs x 1 x memory_size
    remove_embedding = tf.tile(remove_embedding, [batch_size, 1, 1])

    # bs x 1 + time_step x memory_size
    extend_cnx_src = tf.concat([remove_embedding, cnx_src], axis=1)
    extend_src_lengths = src_lengths + 1

    # bs x 1 + time_step x memory_size
    extend_cnx_tgt = tf.concat([remove_embedding, cnx_tgt], axis=1)
    extend_tgt_lengths = tgt_lengths + 1

    assert d_model % num_heads == 0
    d_small = int(d_model // num_heads)

    st_mha = MultiHeadAttention(num_heads, d_model, d_small, use_dropout, dropout_keep, name='src_tgt_attn')
    ts_mha = MultiHeadAttention(num_heads, d_model, d_small, use_dropout, dropout_keep, name='tgt_src_attn')

    attn_src_tgt = st_mha([cnx_src, extend_cnx_tgt, extend_cnx_tgt, extend_tgt_lengths])  # bs x src_seq_len x word_dim
    attn_tgt_src = ts_mha([cnx_tgt, extend_cnx_src, extend_cnx_src, extend_src_lengths])  # bs x tgt_seq_len x word_dim

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
        wa_inserted_last, wa_deleted_last = wa_accumulator(insert_words, delete_words,
                                                           iw_lengths, dw_lengths, wa_hidden_dim)

        if use_dropout and dropout_keep < 1.:
            wa_inserted_last = tf.nn.dropout(wa_inserted_last, dropout_keep)
            wa_deleted_last = tf.nn.dropout(wa_deleted_last, dropout_keep)

        cnx_encoder = tf.make_template('cnx_encoder', context_encoder,
                                       hidden_dim=ctx_hidden_dim,
                                       num_layers=ctx_hidden_layer,
                                       swap_memory=swap_memory,
                                       use_dropout=use_dropout,
                                       dropout_keep=dropout_keep)

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

        return edit_vector, (cnx_src, micro_evs_st), (cnx_tgt, micro_evs_ts)
