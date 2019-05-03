from models.common.vocab import PAD_TOKEN
from models.neural_editor.input import convert_to_bytes, get_generator

DATASET_CACHE = {}

import random

import tensorflow as tf
from tqdm import tqdm

tf.logging.set_verbosity(tf.logging.INFO)

from models.common import vocab, util


def create_oov(words):
    word2id = vocab.get_vocab_lookup_tables()[vocab.RAW_WORD2ID]

    ids = []
    ids_extended = []
    oov = []

    for w in words:
        if w not in word2id:
            if w not in oov:
                oov.append(w)

            ids_extended.append(len(word2id) + oov.index(w))
            ids.append(word2id[vocab.UNKNOWN_TOKEN])
        else:
            ids.append(word2id[w])
            ids_extended.append(word2id[w])

    return ids, ids_extended, oov


def words2ids(words, oov=None):
    if oov is None:
        oov = []

    word2id = vocab.get_vocab_lookup_tables()[vocab.RAW_WORD2ID]

    ids = []
    for w in words:
        if w not in word2id:
            if w in oov:
                ids.append(len(word2id) + oov.index(w))
            else:
                ids.append(word2id[vocab.UNKNOWN_TOKEN])
        else:
            ids.append(word2id[w])

    return ids


def infer_dtype(lst):
    if len(lst) > 0:
        item = lst[0]
        if isinstance(item, int):
            dtype = tf.int64
        elif isinstance(item, str):
            dtype = tf.string
        else:
            dtype = tf.string
    else:
        dtype = tf.string

    return dtype


def parse_instance(instance, noiser=None, free=None):
    if isinstance(instance, str):
        instance = instance.split('\t')

    assert len(instance) == 4

    base, output = instance[:2]
    base_words = base.lower().split(' ')
    output_words = output.lower().split(' ')

    orig_base_ids, extended_base_ids, oov = create_oov(base_words)
    orig_output_ids, extended_output_ids = words2ids(output_words), words2ids(output_words, oov)

    src, tgt = instance[2:]
    src_words = src.lower().split(' ')
    tgt_words = tgt.lower().split(' ')

    if free is None:
        free = set()

    insert_words = sorted(set(tgt_words) - set(src_words) - free)
    delete_words = sorted(set(src_words) - set(tgt_words) - free)

    if noiser:
        src_words, tgt_words, insert_words, delete_words = noiser(
            (src_words, tgt_words, insert_words, delete_words)
        )

    if len(insert_words) == 0:
        insert_words.append(vocab.UNKNOWN_TOKEN)

    if len(delete_words) == 0:
        delete_words.append(vocab.UNKNOWN_TOKEN)

    return (orig_base_ids, extended_base_ids,
            orig_output_ids, extended_output_ids,
            words2ids(src_words), words2ids(tgt_words),
            words2ids(insert_words), words2ids(delete_words),
            convert_to_bytes(oov))


def read_examples_from_file(file_path, num_samples=None, seed=0, noiser=None, free_set=None):
    print("new input")
    if not isinstance(file_path, str):
        file_path = str(file_path)

    if file_path in DATASET_CACHE:
        print('Reading examples from cache...')
        examples = DATASET_CACHE[file_path]
    else:
        print('Reading examples from %s...' % file_path)
        with open(file_path, encoding='utf8') as f:
            lines = map(lambda x: x[:-1], f)
            examples = map(lambda x: parse_instance(x, noiser, free_set), lines)
            examples = list(tqdm(examples, total=util.get_num_total_lines(file_path)))

        DATASET_CACHE[file_path] = examples

    if num_samples and len(examples) > num_samples:
        random.seed(seed)
        examples = random.sample(examples, num_samples)

    return examples


def input_fn(file_path, vocab_table, batch_size, num_epochs=None, num_examples=None, seed=0, noiser=None,
             use_free_set=False, shuffle_input=True):
    vocab_table = vocab.get_vocab_lookup_tables()[vocab.STR_TO_INT]

    pad_token = tf.constant(bytes(PAD_TOKEN, encoding='utf8'), dtype=tf.string)
    pad_value = vocab_table.lookup(pad_token)

    base_dataset = read_examples_from_file(
        file_path, num_examples, seed,
        noiser, util.get_free_words_set() if use_free_set else None
    )

    dataset_splits = []
    for index in range(len(base_dataset[0])):
        split_dtype = infer_dtype(base_dataset[0][index])

        split = tf.data.Dataset.from_generator(
            generator=get_generator(base_dataset, index),
            output_types=(split_dtype),
            output_shapes=(None,)
        )

        if split_dtype == tf.string:
            pad = pad_token
        else:
            pad = pad_value

        split = split.padded_batch(
            batch_size,
            padded_shapes=[None],
            padding_values=pad
        )

        dataset_splits.append(split)

    dataset = tf.data.Dataset.zip(tuple(dataset_splits))
    if num_epochs and shuffle_input:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(500, num_epochs))
    elif num_epochs:
        dataset = dataset.repeat(num_epochs)

    fake_label = tf.data.Dataset.from_tensor_slices(tf.constant([0])).repeat()

    dataset = dataset.zip((dataset, fake_label)) \
        .prefetch(1)

    return dataset
