import tensorflow as tf
from models.common.metrics import join_tokens, create_bleu_metric_ops


def test_join_prediction():
    with tf.Graph().as_default():
        tokens = [
            [b'11', b'12', b'13', b'14', b'15'],
            [b'21', b'22', b'p', b'p', b'p'],
            [b'31', b'32', b'33', b'p', b'p'],
        ]
        token_lengths = [5, 2, 3]

        tokens = tf.constant(tokens, dtype=tf.string)
        token_lengths = tf.constant(token_lengths)

        joined = join_tokens(tokens, token_lengths, separator=' ')
        with tf.Session() as sess:
            print(sess.run(joined))


def test_zero_one():
    ref = [
        [b'41', b'12', b'13', b'14', b'15', b'<stop>'],
        [b'21', b'22', b'<stop>', b'p', b'p', b'p'],
        [b'<start>', b'31', b'62', b'33', b'<stop>', b'p'],
    ]
    ref_lengths = [6, 3, 5]

    pred = [
        [b'11', b'12', b'13', b'14', b'15', b'16'],
        [b'<start>', b'21', b'22', b'p', b'p', b'p'],
        [b'31', b'32', b'33', b'18', b'p', b'p'],
    ]
    pred_lengths = [6, 5, 4]

    ref = tf.constant(ref, dtype=tf.string)
    pred = tf.constant(pred, dtype=tf.string)
    ref_lengths = tf.constant(ref_lengths)
    pred_lengths = tf.constant(pred_lengths)

    blues = create_bleu_metric_ops(ref, pred, ref_lengths, pred_lengths)

    with tf.Session() as sess:
        print()
        print(sess.run(blues))
