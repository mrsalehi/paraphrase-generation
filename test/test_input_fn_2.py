import os

import tensorflow as tf

tf.enable_eager_execution()
from models.im_attn_ee_rnn_attn_dec_copy_net.input import input_fn
from models.common import vocab, sequence

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

NUM_DATASET_EXAMPLE = 6
BATCH_SIZE = 2
NUM_EPOCH = 5

MIN_SENT_LEN = 4
MAX_SENT_LEN = 7


def test_num_examples():
    V, embed_matrix = vocab.read_word_embeddings(
        'data/word_vectors/glove.6B.300d_dbpedia.txt',
        300,
        10000
    )

    vocab_table = vocab.create_vocab_lookup_tables(V)
    dataset = input_fn('data/quora_naug/train.tsv', vocab_table, 10, 1)
    for (features, _) in dataset:
        base_words, extended_base_words, \
        output_words, extended_output_words, \
        src_words, tgt_words, \
        inserted_words, deleted_words, \
        oov = features

        oov_len = sequence.length_string(oov, vocab.PAD_TOKEN)

        print(features)
