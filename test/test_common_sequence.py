import random
import numpy as np
import tensorflow as tf
import os
from models.common.sequence import length, length_pre_embedding

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

EMBED_DIM = 2
BATCH_SIZE = 4
MAX_LEN = 10


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
