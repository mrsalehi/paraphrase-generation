import tensorflow as tf
import tensorflow_probability as tfp

from models.im_rnn_enc.edit_encoder import context_encoder

tfd = tfp.distributions

from models.neural_editor.edit_encoder import sample_vMF

import models.common.sequence as sequence

OPS_NAME = 'edit_encoder'


def rnn_encoder(base_sent_embed, source_words, target_words, insert_words, delete_words,
                source_lengths, target_lengths, iw_lengths, dw_lengths,
                ctx_hidden_dim, ctx_hidden_layer, wa_hidden_dim, wa_hidden_layer,
                edit_dim, noise_scaler, norm_eps, norm_max, sent_encoder,
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
        cnx_src, cnx_src_last = sent_encoder(source_words, source_lengths)
        cnx_tgt, cnx_tgt_last = sent_encoder(target_words, target_lengths)

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

        features = tf.concat([
            cnx_src_last,
            cnx_tgt_last,
            wa_inserted_last,
            wa_deleted_last
        ], axis=1)

        candidate_edit_vector = tf.layers.dense(features, edit_dim, use_bias=False, name='encoder_ev')

        gate = tf.layers.dense(
            tf.concat([base_sent_embed, cnx_src_last], axis=1),
            units=edit_dim,
            activation='sigmoid', use_bias=True,
            name='ee_gate'
        )

        edit_vector = gate * candidate_edit_vector

        if enable_vae:
            edit_vector = sample_vMF(edit_vector, noise_scaler, norm_eps, norm_max)

        return edit_vector, wa_inserted, wa_deleted
