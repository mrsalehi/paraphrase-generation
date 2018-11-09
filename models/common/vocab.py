import tensorflow as tf
import tensorflow.contrib.lookup as lookup
import numpy as np

PAD_TOKEN = '<pad>'
UNKNOWN_TOKEN = '<unk>'
START_TOKEN = '<start>'
STOP_TOKEN = '<stop>'

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNKNOWN_TOKEN,
    START_TOKEN,
    STOP_TOKEN
]

OOV_TOKEN_ID = 0


def emulate_distribution(shape, target_samples):
    m = np.mean(target_samples)
    s = np.std(target_samples)
    samples = np.random.normal(m, s, size=shape)

    return samples


def get_special_tokens_embeds(embeddings):
    pad_embedding = np.zeros((1, embeddings.shape[1]), dtype=np.float32)

    shape = (len(SPECIAL_TOKENS) - 1, embeddings.shape[1])
    special_embeddings = emulate_distribution(shape, embeddings)
    special_embeddings = special_embeddings.astype(np.float32)

    return np.concatenate([pad_embedding, special_embeddings], axis=0)


def read_word_embeddings(file_path, embed_dim,
                         vocab_size=None,
                         include_special_tokens=True,
                         special_tokens=None):
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS

    embeds = []
    vocab = []
    if include_special_tokens:
        vocab += special_tokens

    with open(file_path, encoding='utf8') as f:
        for i, line in enumerate(f):
            if vocab_size and i == vocab_size:
                break

            line = line[:-1]
            tokens = line.split(' ')

            word, embed = tokens[0], np.array([float(t) for t in tokens[1:]])
            assert len(embed) == embed_dim

            vocab.append(word)
            embeds.append(embed)

    if vocab_size is None:
        vocab_size = len(embeds)

    embedding_matrix = np.stack(embeds)
    assert embedding_matrix.shape == (vocab_size, embed_dim)

    if include_special_tokens:
        special_token_embeds = get_special_tokens_embeds(embedding_matrix)
        embedding_matrix = np.concatenate([special_token_embeds, embedding_matrix], axis=0)
        assert embedding_matrix.shape == (vocab_size + len(SPECIAL_TOKENS), embed_dim)

    return vocab, embedding_matrix


def get_vocab_lookup(vocab, name=None, reuse=None):
    with tf.variable_scope(name, 'vocab_lookup', reuse=reuse):
        vocab_lookup = lookup.index_table_from_tensor(
            mapping=vocab,
            num_oov_buckets=0,
            default_value=OOV_TOKEN_ID,
            name=name
        )

    return vocab_lookup


def init_embeddings(embed_matrix):
    with tf.variable_scope('embedding_lookup', reuse=False):
        embeddings = tf.get_variable('embeddings',
                                     shape=embed_matrix.shape,
                                     initializer=tf.constant_initializer(embed_matrix),
                                     trainable=True)

    return embeddings


def embed_tokens(ids):
    with tf.variable_scope('embedding_lookup', reuse=True):
        embeddings = tf.get_variable('embeddings')

    return tf.nn.embedding_lookup(embeddings, ids)
