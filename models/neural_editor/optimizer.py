import tensorflow as tf
from tensorflow.contrib import seq2seq

from models.neural_editor import decoder

OPS = 'optimization'


def loss(dec_output, gold_rnn_output, lengths):
    rnn_output = decoder.rnn_output(dec_output)
    with tf.name_scope('optimization'):
        batch_size = tf.shape(lengths)
        mask = tf.sequence_mask(lengths, dtype=tf.float32)

        # [batch x max_len]
        final_loss = seq2seq.sequence_loss(
            rnn_output,
            gold_rnn_output,
            weights=mask,
            average_across_timesteps=False,
            average_across_batch=False
        )

        final_loss = tf.reduce_sum(final_loss, axis=1)
        op = tf.assert_equal(tf.shape(final_loss), batch_size)
        with tf.control_dependencies([op]):
            final_loss = tf.reduce_mean(final_loss)

    return final_loss


def train(loss, lr, num_steps_to_observe_norm):
    with tf.name_scope('optimization'):
        global_step = tf.train.get_global_step()

        optimizer = tf.train.AdamOptimizer(lr)
        gradients = optimizer.compute_gradients(loss)

        grads = [grad for grad, var in gradients]
        vars = [var for grad, var in gradients]

        max_grad_norm_var = tf.get_variable('max_grad_norm', shape=(), dtype=tf.float32,
                                            initializer=tf.zeros_initializer())

        def true():
            clipped, global_norm = tf.clip_by_global_norm(grads, 1e10)
            asg = tf.assign(max_grad_norm_var, tf.maximum(max_grad_norm_var, 2. * global_norm))
            with tf.control_dependencies([asg]):
                global_norm = tf.identity(global_norm)

            return clipped, global_norm

        def false():
            return tf.clip_by_global_norm(grads, max_grad_norm_var)

        clipped, current_global_norm = tf.cond(global_step < num_steps_to_observe_norm, true, false)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            tf.logging.info("Update ops")
            tf.logging.info(update_ops)
            train_op = optimizer.apply_gradients(zip(clipped, vars), global_step)

        return train_op, current_global_norm
