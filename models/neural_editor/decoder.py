import tensorflow as tf


def prepare_decoder_inputs(target_words, start_token_id):
    batch_size = tf.shape(target_words)[0]
    start_tokens = tf.fill([batch_size, 1], start_token_id)
    inputs = tf.concat([start_tokens, target_words], axis=1)

    return inputs


def prepare_decoder_output(target_words, lengths, stop_token_id, pad_token_id):
    batch_size = tf.shape(target_words)[0]

    extra_pad = tf.fill([batch_size, 1], pad_token_id)
    target_words = tf.concat([target_words, extra_pad], axis=1)
    max_length = tf.shape(target_words)[1]

    update_indices = tf.range(0, batch_size) * max_length + (lengths)
    update_indices = tf.reshape(update_indices, [-1, 1])
    flatten = tf.reshape(target_words, [-1])

    updates = tf.fill([batch_size], stop_token_id)
    delta = tf.scatter_nd(update_indices, updates, tf.shape(flatten))

    outputs = flatten + delta
    outputs = tf.reshape(outputs, [-1, max_length])

    return outputs
