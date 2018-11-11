import tensorflow as tf
import tensorflow.contrib.rnn as tf_rnn


def length(sequence, max_axis=2, name=None):
    with tf.name_scope(name, 'seq_length', [sequence]):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), max_axis))
        length = tf.reduce_sum(used, max_axis - 1)
        length = tf.cast(length, tf.int32)

    return length


def length_pre_embedding(sequence, name=None):
    with tf.name_scope(name, 'seq_length', [sequence]):
        used = tf.sign(sequence)
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)

    return length


def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


def create_trainable_initial_states(batch_size, cell, name_prefix='zero_states'):
    if not cell._state_is_tuple:
        initial_state = tf.get_variable(
            name_prefix,
            [1, cell.state_size],
            initializer=tf.constant_initializer(0.0),
            trainable=True)
        return tf.tile(initial_state, [batch_size, 1])

    initial_variables = []
    for i, (c, h) in enumerate(cell.state_size):
        init_c = tf.get_variable(
            '%s_c_%s' % (name_prefix, i),
            [1, c],
            initializer=tf.constant_initializer(0.0),
            trainable=True)

        init_h = tf.get_variable(
            '%s_h_%s' % (name_prefix, i),
            [1, h],
            initializer=tf.constant_initializer(0.0),
            trainable=True
        )

        initial_variables.append(tf_rnn.LSTMStateTuple(
            tf.tile(init_c, [batch_size, 1]),
            tf.tile(init_h, [batch_size, 1])
        ))

    return tuple(initial_variables)
