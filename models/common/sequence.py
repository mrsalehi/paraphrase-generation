import tensorflow as tf
import tensorflow.contrib.rnn as tf_rnn
from tensorflow.contrib import seq2seq


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


def length_string(sequence, pad, name=None):
    with tf.name_scope(name, 'seq_length', [sequence]):
        used = tf.not_equal(sequence, pad)
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


def get_tiled_variable(batch_size, beam_width, **kwargs):
    init_variable = tf.get_variable(**kwargs)
    init_variable = tf.tile(init_variable, [batch_size, 1], name='tiled_%s' % (kwargs['name']))
    if beam_width:
        init_variable = seq2seq.tile_batch(init_variable, beam_width, name='tiled_%s' % (kwargs['name']))

    return init_variable


def create_trainable_lstm_initial_state(state_size, batch_size, name_prefix, beam_width=None):
    c, h = state_size
    return tf_rnn.LSTMStateTuple(
        get_tiled_variable(batch_size,
                           beam_width,
                           name='%sc' % name_prefix,
                           shape=[1, c],
                           initializer=tf.constant_initializer(0.0),
                           trainable=True),
        get_tiled_variable(batch_size,
                           beam_width,
                           name='%sh' % name_prefix,
                           shape=[1, h],
                           initializer=tf.constant_initializer(0.0),
                           trainable=True)
    )


def create_trainable_initial_states(batch_size, cell, name_prefix='zero_states'):
    if not cell._state_is_tuple:
        initial_state = tf.get_variable(
            name_prefix,
            [1, cell.state_size],
            initializer=tf.constant_initializer(0.0),
            trainable=True)
        return tf.tile(initial_state, [batch_size, 1])

    initial_variables = []
    for i, size in enumerate(cell.state_size):
        initial_variables.append(
            create_trainable_lstm_initial_state(size, batch_size, '%s_%s_' % (name_prefix, i))
        )

    return tuple(initial_variables)


def create_trainable_initial_states_ss(batch_size, state_size, name_prefix='zero_states', beam_width=None):
    initial_variables = []
    for i, size in enumerate(state_size):
        initial_variables.append(
            create_trainable_lstm_initial_state(size, batch_size, '%s_%s_' % (name_prefix, i), beam_width)
        )

    return tuple(initial_variables)
