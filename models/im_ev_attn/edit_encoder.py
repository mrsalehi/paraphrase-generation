import tensorflow as tf

import models.common.sequence as sequence
from models.im_rnn_enc.edit_encoder import word_aggregator, context_encoder
from models.neural_editor.edit_encoder import sample_vMF

OPS_NAME = 'edit_encoder'


def rnn_encoder(source_words, target_words, insert_words, delete_words,
                source_lengths, target_lengths, iw_lengths, dw_lengths,
                ctx_hidden_dim, ctx_hidden_layer, wa_hidden_dim, wa_hidden_layer,
                edit_dim, noise_scaler, norm_eps, norm_max, dropout_keep=1., use_dropout=False,
                swap_memory=False, enable_vae=True):
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

        cnx_src = cnx_encoder(source_words, source_lengths)
        cnx_tgt = cnx_encoder(target_words, target_lengths)

        cnx_src_last = sequence.last_relevant(cnx_src, source_lengths)
        cnx_tgt_last = sequence.last_relevant(cnx_tgt, target_lengths)

        if use_dropout and dropout_keep < 1.:
            cnx_src_last = tf.nn.dropout(cnx_src_last, dropout_keep)
            cnx_tgt_last = tf.nn.dropout(cnx_tgt_last, dropout_keep)

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

        features = tf.concat([
            cnx_src_last,
            cnx_tgt_last,
            wa_inserted_last,
            wa_deleted_last
        ], axis=1)

        edit_vector = tf.layers.dense(features, edit_dim, name='encoder_ev')

        if enable_vae:
            edit_vector = sample_vMF(edit_vector, noise_scaler, norm_eps, norm_max)

        return edit_vector, wa_inserted, wa_deleted
