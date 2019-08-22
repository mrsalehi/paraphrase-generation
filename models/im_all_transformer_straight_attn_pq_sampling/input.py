import numpy as np

from models.common.vocab import PAD_TOKEN
from models.im_all_transformer.input import read_examples_from_file

DATASET_CACHE = {}

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

from models.common import vocab, util


def input_fn(file_path, vocab_table, config, batch_size, num_epochs=None, num_examples=None, seed=0, noiser=None,
             use_free_set=False, shuffle_input=True):
    gen = read_examples_from_file(
        file_path, config, num_examples, seed,
        noiser, util.get_free_words_set() if use_free_set else None, return_gen=True
    )

    probs = util.load_str_list(str(file_path) + '_probs')
    probs = [float(p) for p in probs]
    dataset_probs = tf.data.Dataset.from_tensor_slices(
        np.array(probs, dtype=np.float32).reshape((-1, 1)))
    dataset_probs = dataset_probs.batch(batch_size)

    vocab_table = vocab.get_vocab_lookup_tables()[vocab.STR_TO_INT]

    pad_id = tf.constant(vocab.SPECIAL_TOKENS.index(PAD_TOKEN), dtype=tf.int64)

    dataset = tf.data.Dataset.from_generator(
        generator=gen,
        output_types=(tf.string, tf.string, tf.string, tf.string, tf.string, tf.string),
        output_shapes=(tf.TensorShape([None]), tf.TensorShape([None]),
                       tf.TensorShape([None]), tf.TensorShape([None]),
                       tf.TensorShape([None]), tf.TensorShape([None]))
    )
    dataset = dataset.map(lambda *x: tuple([vocab_table.lookup(i) for i in x]))
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None]),
                       tf.TensorShape([None]), tf.TensorShape([None]),
                       tf.TensorShape([None]), tf.TensorShape([None])),
        padding_values=tuple([pad_id] * 6)
    )

    dataset = tf.data.Dataset.zip((dataset, dataset_probs))

    if num_epochs and shuffle_input:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(500, num_epochs))
    elif num_epochs:
        dataset = dataset.repeat(num_epochs)

    fake_label = tf.data.Dataset.from_tensor_slices(tf.constant([0])).repeat()

    dataset = dataset.zip((dataset, fake_label))
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return dataset
