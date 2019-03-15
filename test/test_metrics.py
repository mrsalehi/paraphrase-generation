import random

import numpy as np
import tensorflow as tf

from models.common.metrics import join_tokens, create_bleu_metric_ops, join_beams


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


def test_join_beam_tokens():
    def generate_beam(min_val=10, max_val=50, useless_val=88, min_len=5, max_len=10, beam_size=2):
        seqs = []
        lengths = []
        for _ in range(beam_size):
            seq_len = random.randint(min_len, max_len)
            seq = [bytes(str(random.randint(min_val, max_val)), encoding='utf8') for _ in range(seq_len)]
            seq += [bytes(str(useless_val), encoding='utf8')] * (max_len - seq_len)

            seqs.append(seq)
            lengths.append(seq_len)

        seqs = list(zip(*seqs))

        return seqs, lengths

    batch_size = 3

    with tf.Graph().as_default():
        tokens, lengths = zip(*[generate_beam() for _ in range(batch_size)])
        tokens = np.array(tokens)
        tokens = tf.constant(tokens, dtype=tf.string)
        lengths = tf.constant(lengths)

        o = join_tokens(tokens, lengths)
        o = join_beams(tokens, lengths)

        with tf.Session() as sess:
            print(sess.run(o))


def test_zero_one():
    def generate_beam(min_val=10, max_val=50, useless_val=88, min_len=5, max_len=10, beam_size=2):
        seqs = []
        lengths = []
        for _ in range(beam_size):
            seq_len = random.randint(min_len, max_len)
            seq = [bytes(str(random.randint(min_val, max_val)), encoding='utf8') for _ in range(seq_len)]
            seq += [bytes(str(useless_val), encoding='utf8')] * (max_len - seq_len)

            seqs.append(seq)
            lengths.append(seq_len)

        seqs = list(zip(*seqs))

        return seqs, lengths

    batch_size = 3

    ref = [
        [b'41', b'12', b'13', b'14', b'15', b'<stop>'],
        [b'21', b'22', b'<stop>', b'p', b'p', b'p'],
        [b'<start>', b'31', b'62', b'33', b'<stop>', b'p'],
    ]
    ref_lengths = [6, 3, 5]

    pred = [
        [[b'11', b'12', b'13', b'14', b'15', b'16'], [b'11', b'12', b'13', b'14', b'15', b'16']],
        [[b'<start>', b'21', b'22', b'p', b'p', b'p'], [b'<start>', b'21', b'22', b'p', b'p', b'p']],
        [[b'31', b'32', b'33', b'18', b'p', b'p'], [b'31', b'32', b'33', b'18', b'p', b'p']],
    ]
    pred_lengths = [[6, 6], [5, 5], [4, 4]]

    ref = tf.constant(ref, dtype=tf.string)
    pred = tf.constant(pred, dtype=tf.string)
    pred = tf.transpose(pred, [0, 2, 1])
    ref_lengths = tf.constant(ref_lengths)
    pred_lengths = tf.constant(pred_lengths)

    best_pred = pred_lengths[:, :, 0]

    blues = create_bleu_metric_ops(ref, pred, ref_lengths, pred_lengths)

    with tf.Session() as sess:
        print()
        print(sess.run(blues))
