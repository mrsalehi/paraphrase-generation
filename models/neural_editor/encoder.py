import tensorflow as tf
import tensorflow.contrib.rnn as tf_rnn

OPS_NAME = 'source_sentence_encoder'


def bidirectional_encoder(src, src_length,
                          hidden_dim, num_layer,
                          dropout_keep, use_dropout=False, reuse=None, name=None):
    with tf.variable_scope(name, 'encoder', values=[src, src_length], reuse=reuse):
        def create_rnn_layer(layer_num, dim):
            cell = tf_rnn.LSTMCell(dim, name='layer_%s' % layer_num)
            cell = tf_rnn.ResidualWrapper(cell)
            return cell

        batch_size = tf.shape(src)[0]

        fw_cell = tf_rnn.MultiRNNCell([create_rnn_layer(i, hidden_dim // 2) for i in range(num_layer)])
        bw_cell = tf_rnn.MultiRNNCell([create_rnn_layer(i, hidden_dim // 2) for i in range(num_layer)])

        fw_zero_state = fw_cell.zero_state(batch_size, tf.float32)
        bw_zero_state = bw_cell.zero_state(batch_size, tf.float32)

        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            src,
            src_length,
            fw_zero_state,
            bw_zero_state
        )

        output = tf.concat(outputs, axis=2)
        assert output.shape[2] == hidden_dim

    return output


def source_sent_encoder(src, src_length,
                        hidden_dim, num_layer,
                        dropout_keep, use_dropout=False, reuse=None):
    return bidirectional_encoder(
        src, src_length,
        hidden_dim, num_layer,
        dropout_keep, use_dropout, reuse, name=OPS_NAME
    )
