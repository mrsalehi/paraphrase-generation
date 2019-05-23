import tensorflow as tf

from models.neural_editor import decoder

OPS = 'optimization'


def loss(dec_output, gold_rnn_output, lengths):
    rnn_output = decoder.rnn_output(dec_output)
    with tf.name_scope('optimization'):
        batch_size = tf.shape(lengths)[0]
        max_dec_len = tf.shape(rnn_output)[1]

        mask = tf.sequence_mask(lengths, dtype=tf.float32)

        i1, i2 = tf.meshgrid(tf.range(batch_size), tf.range(max_dec_len), indexing="ij")
        indices = tf.stack((tf.to_int64(i1), tf.to_int64(i2), gold_rnn_output), axis=2)

        probs = tf.gather_nd(rnn_output, indices)
        probs = tf.where(tf.less_equal(probs, 0), tf.ones_like(probs) * 1e-10, probs)

        crossent = -tf.log(probs)
        final_loss = tf.reduce_sum(crossent * mask) / tf.to_float(batch_size)

        extra_losses = tf.losses.get_losses()
        extra_losses = sum(extra_losses)

        final_loss = final_loss + extra_losses

    return final_loss


def add_reconst_loss(original, reconstructed):
    diff = tf.squared_difference(original, reconstructed)
    loss = tf.reduce_sum(diff, axis=1)
    loss = tf.reduce_mean(loss)

    tf.losses.add_loss(loss)

    return loss
