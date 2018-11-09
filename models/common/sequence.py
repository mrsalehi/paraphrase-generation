import tensorflow as tf


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
