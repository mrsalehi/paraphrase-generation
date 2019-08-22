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


def _pad_tensors_to_same_length(x, y):
    """Pad x and y so that the results have the same length (second dimension)."""
    with tf.name_scope("pad_to_same_length"):
        x_length = tf.shape(x)[1]
        y_length = tf.shape(y)[1]

        max_length = tf.maximum(x_length, y_length)

        x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])
        y = tf.pad(y, [[0, 0], [0, max_length - y_length]])
        return x, y


def padded_cross_entropy_loss(logits, labels, label_lengths, loss_weights, smoothing, vocab_size):
    """Calculate cross entropy loss while ignoring padding.

    Args:
      logits: Tensor of size [batch_size, length_logits, vocab_size]
      labels: Tensor of size [batch_size, length_labels]
      smoothing: Label smoothing constant, used to determine the on and off values
      vocab_size: int size of the vocabulary
    Returns:
      Returns the cross entropy loss and weight tensors: float32 tensors with
        shape [batch_size, max(length_logits, length_labels)]
    """
    with tf.name_scope("loss", values=[logits, labels]):
        logits, labels = _pad_tensors_to_same_length(logits, labels)

        # Calculate smoothing cross entropy
        with tf.name_scope("smoothing_cross_entropy", values=[logits, labels]):
            confidence = 1.0 - smoothing
            low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)
            soft_targets = tf.one_hot(
                tf.cast(labels, tf.int32),
                depth=vocab_size,
                on_value=confidence,
                off_value=low_confidence)
            xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=soft_targets)

            # Calculate the best (lowest) possible value of cross entropy, and
            # subtract from the cross entropy loss.
            normalizing_constant = -(
                    confidence * tf.log(confidence) + tf.to_float(vocab_size - 1) *
                    low_confidence * tf.log(low_confidence + 1e-20))
            xentropy -= normalizing_constant

        assert len(loss_weights.shape) == 2

        weights = tf.to_float(tf.not_equal(labels, 0))
        xentropy = xentropy * loss_weights * weights
        loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

        return loss, weights


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    with tf.name_scope("learning_rate"):
        warmup_steps = tf.to_float(learning_rate_warmup_steps)
        step = tf.to_float(tf.train.get_or_create_global_step())

        learning_rate *= (hidden_size ** -0.5)
        # Apply linear warmup
        learning_rate *= tf.minimum(1.0, step / warmup_steps)
        # Apply rsqrt decay
        learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))

        tf.summary.scalar("learning_rate", learning_rate, collections=['extra'])

        return learning_rate


def get_train_op(loss, config):
    with tf.variable_scope("get_train_op"):
        learning_rate = get_learning_rate(
            learning_rate=config.optim.learning_rate,
            hidden_size=config.editor.transformer.hidden_size,
            learning_rate_warmup_steps=config.optim.learning_rate_warmup_steps)

        # Create optimizer. Use LazyAdamOptimizer from TF contrib, which is faster
        # than the TF core Adam optimizer.
        optimizer = tf.contrib.opt.LazyAdamOptimizer(
            learning_rate,
            beta1=config.optim.adam_beta1,
            beta2=config.optim.adam_beta2,
            epsilon=config.optim.adam_epsilon)

        # Uses automatic mixed precision FP16 training if on GPU.
        if config.get('optim.enable_mixed_precision', False):
            optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
                optimizer)

        # Calculate and apply gradients using LazyAdamOptimizer.
        global_step = tf.train.get_global_step()
        tvars = tf.trainable_variables()
        gradients = optimizer.compute_gradients(loss, tvars, colocate_gradients_with_ops=True)
        minimize_op = optimizer.apply_gradients(gradients, global_step=global_step, name="train")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)

        gradient_norm = tf.global_norm(list(zip(*gradients))[0])
        tf.summary.scalar('gradient_norm', gradient_norm, ['extra'])

        return train_op
