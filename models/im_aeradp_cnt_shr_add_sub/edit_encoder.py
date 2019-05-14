import tensorflow as tf

import models.common.sequence as sequence
from models.im_attn_ee.edit_encoder import context_encoder, create_micro_edit_vectors
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


def attn_encoder(source_words, target_words, insert_words, delete_words,
                 source_lengths, target_lengths, iw_lengths, dw_lengths,
                 ctx_hidden_dim, ctx_hidden_layer, wa_hidden_dim, wa_hidden_layer, meve_hidden_dim, meve_hidden_layers,
                 edit_dim, micro_edit_ev_dim, num_heads, noise_scaler, norm_eps, norm_max, sent_encoder,
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

        cnx_src, _ = sent_encoder(source_words, source_lengths)
        cnx_tgt, _ = sent_encoder(target_words, target_lengths)

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
