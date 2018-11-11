import tensorflow as tf
import tensorflow.contrib.rnn as tf_rnn
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

import models.common.sequence as sequence

OPS_NAME = 'edit_encoder'


def word_aggregator(words, lengths, hidden_dim, num_layers, name=None, reuse=None):
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

        def create_rnn_layer(layer_num):
            cell = tf_rnn.LSTMCell(hidden_dim, name='layer_%s' % layer_num)
            return cell

        cell = tf_rnn.MultiRNNCell([create_rnn_layer(i) for i in range(num_layers)])
        zero_state = sequence.create_trainable_initial_states(batch_size, cell)

        outputs, last_state = tf.nn.dynamic_rnn(
            cell,
            words,
            sequence_length=lengths,
            initial_state=zero_state
        )

        return outputs


def context_encoder(words, lengths, hidden_dim, num_layers, name=None, reuse=None):
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
            bw_zero_state
        )

        output = tf.concat(outputs, axis=2)
        assert output.shape[2] == hidden_dim

    return output


def rnn_encoder(source_words, target_words, insert_words, delete_words,
                source_lengths, target_lengths, iw_lengths, dw_lengths,
                ctx_hidden_dim, ctx_hidden_layer, wa_hidden_dim, wa_hidden_layer,
                edit_dim, noise_scaler, norm_eps, norm_max, dropout_keep):
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
                                       num_layers=ctx_hidden_layer)

        cnx_src = cnx_encoder(source_words, source_lengths)
        cnx_tgt = cnx_encoder(target_words, target_lengths)

        cnx_src_last = sequence.last_relevant(cnx_src, source_lengths)
        cnx_tgt_last = sequence.last_relevant(cnx_tgt, target_lengths)

        cnx_src_last = tf.nn.dropout(cnx_src_last, dropout_keep)
        cnx_tgt_last = tf.nn.dropout(cnx_tgt_last, dropout_keep)

        wa = tf.make_template('wa', word_aggregator,
                              hidden_dim=ctx_hidden_dim,
                              num_layers=ctx_hidden_layer)

        wa_inserted = wa(insert_words, iw_lengths)
        wa_deleted = wa(delete_words, dw_lengths)

        wa_inserted_last = sequence.last_relevant(wa_inserted, iw_lengths)
        wa_deleted_last = sequence.last_relevant(wa_deleted, dw_lengths)

        wa_inserted_last = tf.nn.dropout(wa_inserted_last)
        wa_deleted_last = tf.nn.dropout(wa_deleted_last)

        features = tf.concat([
            cnx_src_last,
            cnx_tgt_last,
            wa_inserted_last,
            wa_deleted_last
        ], dim=1)

        edit_vector = tf.layers.dense(features, edit_dim, 'encoder_ev')


def sample_vMF(m, kappa, norm_eps, norm_max):
    batch_size = tf.shape(m)[0]
    id_dim = m.shape[1]

    munorm = tf.norm(m, axis=1, keepdims=True)
    munorm = tf.tile(munorm, [1, id_dim])
    munoise = add_norm_noise(munorm, norm_eps, norm_max, batch_size)

    w = sample_weight_tf(kappa, id_dim, batch_size)
    wtorch = w * tf.ones((batch_size, id_dim))

    v = sample_orthonormal_to(m / munorm, id_dim, batch_size)
    scale_factr = tf.sqrt(tf.ones((batch_size, id_dim)) - tf.pow(wtorch, 2))
    orth_term = v * scale_factr
    muscale = m * wtorch / munorm
    sampled_vec = (orth_term + muscale) * munoise

    return sampled_vec


def sample_orthonormal_to(mu, dim, batch_size):
    v = tf.random_normal(shape=(batch_size, dim))
    rescale_value = tf.reshape(tf.reduce_sum(mu * v, axis=1), [-1, 1])
    rescale_value = rescale_value / tf.norm(mu, axis=1, keepdims=True)
    proj_mu_v = mu * tf.tile(rescale_value, [1, dim])
    ortho = v - proj_mu_v
    ortho_norm = tf.norm(ortho, axis=1, keepdims=True)
    return ortho / tf.tile(ortho_norm, [1, dim])


def sample_weight_tf(kappa, dim, batch_size):
    dim = int(dim) - 1
    b = dim / (np.sqrt(4. * kappa ** 2 + dim ** 2) + 2 * kappa)  # b= 1/(sqrt(4.* kdiv**2 + 1) + 2 * kdiv)
    x = (1. - b) / (1. + b)
    c = kappa * x + dim * np.log(1 - x ** 2)  # dim * (kdiv *x + np.log(1-x**2))

    b_dist = tfd.Beta(dim / 2., dim / 2.)

    z = b_dist.sample([batch_size, 1])
    w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
    u = tf.random_uniform(shape=(batch_size, 1), minval=0, maxval=1)

    def cond(w, u):
        all_cond = kappa * w + dim * tf.log(1. - x * w) - c < tf.log(u)
        return tf.reduce_all(all_cond)

    def body(w, u):
        z = b_dist.sample([batch_size, 1])
        w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
        u = tf.random_uniform(shape=(batch_size, 1), minval=0, maxval=1)
        return (w, u)

    w, u = tf.while_loop(
        cond, body,
        loop_vars=(w, u),
        back_prop=False,
    )

    return w


def add_norm_noise(norm, eps, norm_max, batch_size):
    trand = tf.random_uniform((batch_size, 1), maxval=1, dtype=tf.float32)
    trand = tf.tile(trand, [1, norm.shape[1]])
    trand = trand * eps
    return hardtanh(norm, 0, norm_max - eps) + trand


def hardtanh(x, min_val, max_val):
    lower = tf.maximum(x, min_val)
    upper = tf.minimum(lower, max_val)
    return upper
