import tensorflow as tf

import models.common.sequence as sequence
from models.im_attn_ee.edit_encoder import context_encoder, create_micro_edit_vectors
from models.neural_editor.edit_encoder import sample_vMF

OPS_NAME = 'edit_encoder'


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
        print("Run me!")
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
            aggreg_mev_ts_last
        ], axis=1)

        edit_vector = tf.layers.dense(features, edit_dim, use_bias=False, name='encoder_ev')

        if enable_vae:
            edit_vector = sample_vMF(edit_vector, noise_scaler, norm_eps, norm_max)

        return edit_vector, tf.constant([[0]]), tf.constant([[0]])
