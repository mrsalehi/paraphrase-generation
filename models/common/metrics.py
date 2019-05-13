import numpy as np
import tensorflow as tf

from models.common import vocab


def bleu(reference, predict):
    """Compute sentence-level bleu score.

    Args:
        reference (list[str])
        predict (list[str])
    """
    from nltk.translate import bleu_score

    if len(predict) == 0:
        if len(reference) == 0:
            return 1.0
        else:
            return 0.0

    # TODO(kelvin): is this quite right?
    # use a maximum of 4-grams. If 4-grams aren't present, use only lower n-grams.
    # n = min(4, len(reference), len(predict))
    # if n == 0:
    #     return 0.0
    # weights = tuple([1. / n] * n)  # uniform weight on n-gram precisions
    return bleu_score.sentence_bleu([reference], predict, [0.5, 0.5])


def remove_start_end_tokens(token_lst, token_length, start_token, end_token):
    token_lst = token_lst[:token_length]

    start_index = -1
    end_index = token_length - 1

    for i, t in enumerate(token_lst):
        if t == start_token and i < end_index:
            start_index = i
        if t == end_token and i > start_index:
            end_index = i

    return token_lst[start_index + 1:end_index]


def _compute_blue_batch(ref_tokens, predict_tokens, ref_len, pred_len):
    bleus = []
    for ref, pred, rlen, plen in zip(ref_tokens, predict_tokens, ref_len, pred_len):
        ref = remove_start_end_tokens(
            [_.decode("utf-8") for _ in ref], rlen,
            vocab.START_TOKEN, vocab.STOP_TOKEN
        )
        pred = remove_start_end_tokens(
            [_.decode("utf-8") for _ in pred], plen,
            vocab.START_TOKEN, vocab.STOP_TOKEN
        )

        bleus.append(bleu(ref, pred))

    return np.array(bleus, dtype=np.float32)


def create_bleu_metric_ops(ref_tokens, predict_tokens, ref_len, pred_len):
    return tf.py_func(
        _compute_blue_batch,
        [ref_tokens, predict_tokens, ref_len, pred_len],
        Tout=tf.float32,
        stateful=False
    )


def join_tokens(tokens, tokens_lengths, separator=' '):
    if tokens.shape.ndims > 2:
        return join_tokens_beam(tokens, tokens_lengths, separator)

    token_str_len = tf.strings.length(tokens)
    mask = tf.sequence_mask(tokens_lengths, dtype=tf.int32)
    substr_len = tf.reduce_sum(mask * token_str_len, axis=1) + (tokens_lengths - 1) * len(separator)
    joined = tf.reduce_join(tokens, axis=1, separator=separator)
    final_str = tf.strings.substr(joined,
                                  tf.zeros_like(tokens_lengths),
                                  substr_len)
    final_str = tf.reshape(final_str, [-1, 1])

    return final_str


def join_tokens_beam(tokens, tokens_lengths, separator=' '):
    tokens_lengths = tf.cast(tokens_lengths, tf.int32)

    token_str_len = tf.strings.length(tokens)
    mask = tf.sequence_mask(tokens_lengths, dtype=tf.int32, maxlen=tf.shape(tokens)[1])
    mask = tf.transpose(mask, [0, 2, 1])

    substr_len = tf.reduce_sum(mask * token_str_len, axis=1) + (tokens_lengths - 1) * len(separator)
    joined = tf.reduce_join(tokens, axis=1, separator=separator)
    final_str = tf.strings.substr(joined,
                                  tf.zeros_like(tokens_lengths),
                                  substr_len)

    return final_str


def join_beams(tokens, tokens_lengths, separator='\n'):
    beams = join_tokens_beam(tokens, tokens_lengths)
    joined = tf.reduce_join(beams, axis=1, separator=separator)
    joined = tf.reshape(joined, [-1, 1])
    return joined
