DATASET_CACHE = {}

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

from models.common import vocab


def parse_instance(instance, noiser=None, free=None):
    if isinstance(instance, str):
        instance = instance.split('\t')

    assert len(instance) == 4

    base, output = instance[:2]

    src, tgt = instance[2:]
    src_words = src.lower().split(' ')
    tgt_words = tgt.lower().split(' ')

    if free is None:
        free = set()

    insert_words = sorted(set(tgt_words) - set(src_words) - free)
    delete_words = sorted((set(src_words) - set(tgt_words)) - free)

    if noiser:
        src_words, tgt_words, insert_words, delete_words = noiser(
            (src_words, tgt_words, insert_words, delete_words)
        )

    if len(insert_words) == 0:
        insert_words.append(vocab.UNKNOWN_TOKEN)

    if len(delete_words) == 0:
        delete_words.append(vocab.UNKNOWN_TOKEN)

    return base, \
           output, \
           src, \
           tgt, \
           ' '.join(insert_words), \
           ' '.join(delete_words)
