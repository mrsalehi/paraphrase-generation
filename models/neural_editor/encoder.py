import tensorflow as tf
import tensorflow.contrib.rnn as tf_rnn

import models.common.sequence as sequence

OPS_NAME = 'source_sentence_encoder'


def bidirectional_encoder(src, src_length,
                          hidden_dim, num_layer,
                          dropout_keep, swap_memory=False, use_dropout=False, reuse=None, name=None):
    with tf.variable_scope(name, 'encoder', values=[src, src_length], reuse=reuse):
        def create_rnn_layer(layer_num, dim):
            cell = tf_rnn.LSTMCell(dim, name='layer_%s' % layer_num)
            if use_dropout and dropout_keep < 1.:
                cell = tf_rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep)

            if layer_num > 0:
                cell = tf_rnn.ResidualWrapper(cell)

            return cell

        batch_size = tf.shape(src)[0]

        fw_cell = tf_rnn.MultiRNNCell([create_rnn_layer(i, hidden_dim // 2) for i in range(num_layer)])
        bw_cell = tf_rnn.MultiRNNCell([create_rnn_layer(i, hidden_dim // 2) for i in range(num_layer)])

        fw_zero_state = sequence.create_trainable_initial_states(batch_size, fw_cell, 'fw_zs')
        bw_zero_state = sequence.create_trainable_initial_states(batch_size, bw_cell, 'bw_zs')

        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            src,
            src_length,
            fw_zero_state,
            bw_zero_state,
            swap_memory=swap_memory
        )

        output = tf.concat(outputs, axis=2)
        final_state = sequence.last_relevant(output, src_length)

    return output, final_state


def source_sent_encoder(src, src_length,
                        hidden_dim, num_layer,
                        dropout_keep=1.0, swap_memory=False, use_dropout=False, reuse=None):
    return bidirectional_encoder(
        src, src_length,
        hidden_dim, num_layer, dropout_keep,
        swap_memory, use_dropout, reuse, name=OPS_NAME
    )
