import tensorflow as tf
import tensorflow.contrib.rnn as tf_rnn
import tensorflow_probability as tfp

tfd = tfp.distributions

from models.neural_editor.edit_encoder import sample_vMF

import models.common.sequence as sequence

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

        return edit_vector
