import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import os

from models.common import vocab
from models.common.sequence import length, length_pre_embedding
from models.neural_editor import input_fn, decoder

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

EMBED_DIM = 2
BATCH_SIZE = 4
MAX_LEN = 10


def test_wtf():
    with tf.Graph().as_default():
        V, embed_matrix = vocab.read_word_embeddings(
            Path('../data') / 'word_vectors' / 'glove.6B.300d_yelp.txt',
            300,
            10000
        )

        table = vocab.create_vocab_lookup_tables(V)
        vocab_s2i = table[vocab.STR_TO_INT]
        vocab_i2s = table[vocab.INT_TO_STR]

        dataset = input_fn('../data/yelp_dataset_large_split/train.tsv', table, 64, 1)
        iter = dataset.make_initializable_iterator()

        (src, tgt, iw, dw), _ = iter.get_next()
        src_len = length_pre_embedding(src)
        tgt_len = length_pre_embedding(tgt)
        iw_len = length_pre_embedding(iw)
        dw_len = length_pre_embedding(dw)

        dec_inputs = decoder.prepare_decoder_inputs(tgt, vocab.get_token_id(vocab.START_TOKEN, vocab_s2i))

        dec_output = decoder.prepare_decoder_output(tgt, tgt_len, vocab.get_token_id(vocab.STOP_TOKEN, vocab_s2i),
                                                    vocab.get_token_id(vocab.PAD_TOKEN, vocab_s2i))

        t_src = vocab_i2s.lookup(src)
        t_tgt = vocab_i2s.lookup(tgt)
        t_iw = vocab_i2s.lookup(iw)
        t_dw = vocab_i2s.lookup(dw)

        t_do = vocab_i2s.lookup(dec_output)
        t_di = vocab_i2s.lookup(dec_inputs)

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
            sess.run(iter.initializer)

            while True:
                try:
                    # src, tgt, iw, dw = sess.run([src, tgt, iw, dw])
                    ts, tt, tiw, tdw, tdo, tdi = sess.run([t_src, t_tgt, t_iw, t_dw, t_do, t_di])
                except:
                    break


def test_length():
    def generate_sequence(seq_len):
        seq = []
        for i in range(MAX_LEN):
            if i < seq_len:
                seq.append(np.random.random(size=(EMBED_DIM,)))
            else:
                seq.append(np.zeros(shape=(EMBED_DIM,)))

        return seq

    gold_seq_lengths = [random.randint(4, MAX_LEN) for _ in range(BATCH_SIZE - 1)] + [MAX_LEN]
    sequence_batch = [generate_sequence(l) for l in gold_seq_lengths]

    batch = np.array(sequence_batch, dtype=np.float32)

    tf.enable_eager_execution()
    lengths = length(batch)
    assert lengths.shape == (BATCH_SIZE,)
    assert list(lengths.numpy()) == gold_seq_lengths


def test_length_pre_embedding():
    def generate_sequence(seq_len):
        seq = []
        for i in range(MAX_LEN):
            if i < seq_len:
                seq.append(random.randint(4, 1000))
            else:
                seq.append(0)

        return seq

    gold_seq_lengths = [random.randint(4, MAX_LEN) for _ in range(BATCH_SIZE - 1)] + [MAX_LEN]
    sequence_batch = [generate_sequence(l) for l in gold_seq_lengths]

    batch = np.array(sequence_batch, dtype=np.float32)

    tf.enable_eager_execution()
    lengths = length_pre_embedding(batch)

    assert lengths.shape == (BATCH_SIZE,)
    assert list(lengths.numpy()) == gold_seq_lengths
