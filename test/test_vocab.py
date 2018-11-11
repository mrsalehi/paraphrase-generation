import os

import numpy as np
import tensorflow as tf
import pytest

from models.common import vocab

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

EMBED_DIM = 30
VOCAB = ['a', 'cat', 'sat', 'on', 'the', 'mat', 'practice', 'start', 'trying', 'couple', 'always', 'reliable', 'simple',
         'to', 'project', 'is', 'learning', 'good', 'it', 'any', 'out', 'of', 'baselines.', 'machine']


def generate_embeddings(vocab, embed_dim):
    return np.stack([np.ones(embed_dim) * i * 1.1 for i in range(len(vocab))])


@pytest.fixture(scope='session')
def embedding_file(tmpdir_factory):
    fn = tmpdir_factory.mktemp('data').join('embedding.txt')
    embeddings = generate_embeddings(VOCAB, EMBED_DIM)
    with open(fn, 'w', encoding='utf8') as f:
        for w, embd in zip(VOCAB, embeddings):
            f.write(w + ' ' + ' '.join([str(e) for e in list(embd)]))
            f.write('\n')

    return fn, embeddings


def test_read_word_embeddings_with_special_tokens(embedding_file):
    file_name, gold_embeddings = embedding_file

    cm_vocab, cm_embeddings = vocab.read_word_embeddings(file_name, EMBED_DIM)
    assert cm_vocab == (vocab.SPECIAL_TOKENS + VOCAB)

    for i, embed in enumerate(cm_embeddings[len(vocab.SPECIAL_TOKENS):, :]):
        assert list(gold_embeddings[i]) == list(embed)


def test_read_word_embeddings_without_special_tokens(embedding_file):
    file_name, gold_embeddings = embedding_file

    cm_vocab, cm_embeddings = vocab.read_word_embeddings(file_name, EMBED_DIM, include_special_tokens=False)
    assert cm_vocab == (VOCAB)

    for i, embed in enumerate(cm_embeddings):
        assert list(gold_embeddings[i]) == list(embed)


def test_read_word_embeddings_with_special_tokens_custom_vocab_size(embedding_file):
    file_name, gold_embeddings = embedding_file

    vocab_size = len(VOCAB) // 2
    cm_vocab, cm_embeddings = vocab.read_word_embeddings(file_name, EMBED_DIM, vocab_size)
    assert cm_vocab == (vocab.SPECIAL_TOKENS + VOCAB[:vocab_size])

    for i, embed in enumerate(cm_embeddings[len(vocab.SPECIAL_TOKENS):, :]):
        assert list(gold_embeddings[i]) == list(embed)


def test_read_word_embeddings_without_special_tokens_custom_vocab_size(embedding_file):
    file_name, gold_embeddings = embedding_file

    vocab_size = len(VOCAB) // 2
    cm_vocab, cm_embeddings = vocab.read_word_embeddings(file_name, EMBED_DIM, vocab_size, include_special_tokens=False)
    assert cm_vocab == (VOCAB[:vocab_size])

    for i, embed in enumerate(cm_embeddings):
        assert list(gold_embeddings[i]) == list(embed)


def test_lookup_ops(embedding_file):
    fn, gold_embeds = embedding_file

    cm_vocab, _ = vocab.read_word_embeddings(fn, EMBED_DIM)

    oov_words = ['dsjii', 'disjfi']
    test_strings = vocab.SPECIAL_TOKENS + VOCAB + oov_words

    tf.enable_eager_execution()
    test_strings = tf.constant(test_strings)
    table = vocab.get_vocab_lookup(cm_vocab)
    ids = table.lookup(test_strings)
    assert list(ids.numpy()) == list(range(len(vocab.SPECIAL_TOKENS + VOCAB))) + [vocab.OOV_TOKEN_ID] * len(oov_words)
