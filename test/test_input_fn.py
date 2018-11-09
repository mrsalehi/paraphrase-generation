import os
import copy
import random
import string

import numpy as np
import pytest
import tensorflow as tf
import models.neural_editor as neural_editor
from models.common.vocab import read_word_embeddings, get_vocab_lookup

from test.test_vocab import embedding_file, VOCAB, EMBED_DIM

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

NUM_DATASET_EXAMPLE = 6
BATCH_SIZE = 2
NUM_EPOCH = 5

MIN_SENT_LEN = 4
MAX_SENT_LEN = 7


def generate_embeddings(vocab, embed_dim):
    return np.stack([np.ones(embed_dim) * i * 1.1 for i in range(len(vocab))])


@pytest.fixture(scope='session')
def dataset_file(tmpdir_factory):
    fn = tmpdir_factory.mktemp('data').join('dataset.tsv')

    word_list = [w for w in random.sample(VOCAB, 12)]

    def get_sentence():
        sent_len = random.randint(MIN_SENT_LEN, MAX_SENT_LEN)
        return ['%s' % (w) for w in random.sample(word_list, sent_len)]

    dataset = []
    for i in range(NUM_DATASET_EXAMPLE):
        s1 = ' '.join(get_sentence())
        s2 = ' '.join(get_sentence())

        dataset.append('%s\t%s' % (s1, s2))

    with open(fn, 'w', encoding='utf8') as f:
        for d in dataset:
            f.write(d)
            f.write('\n')

    return fn, dataset


@pytest.fixture(scope='session')
def batches(dataset_file, embedding_file):
    d_fn, gold_dataset = dataset_file
    e_fn, gold_embeds = embedding_file

    with tf.Graph().as_default():
        vocab, _ = read_word_embeddings(e_fn, EMBED_DIM)
        vocab_lookup = get_vocab_lookup(vocab)

        dataset = neural_editor.input_fn(d_fn, vocab_lookup, BATCH_SIZE, NUM_EPOCH)
        iter = dataset.make_initializable_iterator()

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
            sess.run(iter.initializer)

            batches = []
            while True:
                try:
                    batch, label = sess.run(iter.get_next())
                    batches.append(batch)
                except tf.errors.OutOfRangeError:
                    break

    return batches


def test_num_examples(batches):
    num_batch_per_epoch = (NUM_DATASET_EXAMPLE // BATCH_SIZE)
    assert len(batches) == num_batch_per_epoch * NUM_EPOCH


def test_num_epochs(batches):
    num_batch_per_epoch = (NUM_DATASET_EXAMPLE // BATCH_SIZE)
    epochs = [batches[i * num_batch_per_epoch: (i + 1) * num_batch_per_epoch] for i in range(NUM_EPOCH)]
    assert len(epochs) == NUM_EPOCH


def test_batch_size(batches):
    for b in batches:
        assert b.shape[0] == BATCH_SIZE


def test_num_unique_batches(batches):
    def get_batch_strs(bs):
        return [str(b) for b in bs]

    num_batch_per_epoch = (NUM_DATASET_EXAMPLE // BATCH_SIZE)
    batch_strs = get_batch_strs(batches)
    epochs = [get_batch_strs(batches[i * num_batch_per_epoch: (i + 1) * num_batch_per_epoch]) for i in range(NUM_EPOCH)]

    unique_batches = set(batch_strs)
    assert len(unique_batches) == num_batch_per_epoch

    for e in epochs:
        ue = set(e)
        assert len(e) == len(ue)
        assert ue == unique_batches
