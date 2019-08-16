import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.lookup as lookup
from bpemb import BPEmb

from models.common import graph_utils
from models.common.config import Config

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

OOV_TOKEN_ID = 1

VOCAB_LOOKUP_COLL_NAME = 'vocab_lookup'
EMBEDDING_MATRIX_COLL_NAME = 'embedding_matrix'

STR_TO_INT = 'str_to_int'
INT_TO_STR = 'int_to_str'
RAW_WORD2ID = 'word2id'
RAW_ID2WORD = 'id2word'


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
                         special_tokens=None,
                         random_initialization=False,
                         vocab_file=None):
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS

    embeds = []
    vocab = []
    if include_special_tokens:
        vocab += special_tokens

    if not isinstance(file_path, str):
        file_path = str(file_path)

    if vocab_file is not None:
        with open(vocab_file, encoding='utf8') as f:
            custom_vocab = list(map(lambda l: l.strip(), f))
            vocab += custom_vocab

        return vocab, np.random.normal(0, embed_dim ** -0.5, (vocab_size, embed_dim))


    with open(file_path, encoding='utf8') as f:
        for i, line in enumerate(f):
            if vocab_size and i == vocab_size:
                break

            line = line[:-1]
            tokens = line.split(' ')

            word, embed = tokens[0], np.array([float(t) for t in tokens[1:]])
            # assert len(embed) == embed_dim

            vocab.append(word)
            embeds.append(embed)

    if vocab_size is None:
        vocab_size = len(embeds)

    if random_initialization:
        return vocab, np.random.normal(0, embed_dim ** -0.5, (vocab_size, embed_dim))

    embedding_matrix = np.stack(embeds)
    assert embedding_matrix.shape == (vocab_size, embed_dim)

    if include_special_tokens:
        special_token_embeds = get_special_tokens_embeds(embedding_matrix)
        embedding_matrix = np.concatenate([special_token_embeds, embedding_matrix], axis=0)
        assert embedding_matrix.shape == (vocab_size + len(SPECIAL_TOKENS), embed_dim)

    return vocab, embedding_matrix


def read_subword_embeddings(config):
    word_dim = config.editor.word_dim

    bpemb = get_bpemb_instance(config)
    V = [PAD_TOKEN] + bpemb.pieces

    global START_TOKEN
    global STOP_TOKEN
    global SPECIAL_TOKENS

    START_TOKEN = bpemb.BOS_str
    STOP_TOKEN = bpemb.EOS_str

    SPECIAL_TOKENS = [
        PAD_TOKEN,
        UNKNOWN_TOKEN,
        START_TOKEN,
        STOP_TOKEN
    ]

    if not config.editor.get('use_pretrained_embeddings', True):
        embedding_matrix = np.random.normal(0, word_dim ** -0.5, (len(V), word_dim))
        embedding_matrix[0] = np.zeros((word_dim,), dtype=np.float32)
        return V, embedding_matrix

    pad_embedding = np.zeros((1, word_dim), np.float32)
    embedding_matrix = np.concatenate([
        pad_embedding, bpemb.vectors
    ], axis=0)

    assert embedding_matrix.shape == (len(V), word_dim)

    return V, embedding_matrix


def read_t2t_subword_embeddings(config):
    word_dim = config.editor.word_dim

    encoder = get_t2t_subword_encoder_instance(config)
    V = list(encoder.all_subtoken_strings)

    embedding_matrix = np.random.normal(0, word_dim ** -0.5, (len(V), word_dim))
    embedding_matrix[0] = np.zeros((word_dim,), dtype=np.float32)
    return V, embedding_matrix


def get_vocab_lookup(vocab, name=None, reuse=None):
    with tf.variable_scope(name, 'vocab_lookup', reuse=reuse):
        vocab_lookup = lookup.index_table_from_tensor(
            mapping=vocab,
            num_oov_buckets=0,
            default_value=OOV_TOKEN_ID,
            name=name
        )

    return vocab_lookup


def create_vocab_lookup_tables(vocab):
    str_to_int = lookup.index_table_from_tensor(
        mapping=vocab,
        num_oov_buckets=0,
        default_value=OOV_TOKEN_ID,
        name='vocab_lookup_str_to_int'
    )

    int_to_str = lookup.index_to_string_table_from_tensor(
        mapping=vocab,
        default_value=UNKNOWN_TOKEN,
        name='vocab_lookup_int_to_str'
    )

    word2id = {w: i for i, w in enumerate(vocab)}

    vocab_lookup = {
        INT_TO_STR: int_to_str,
        STR_TO_INT: str_to_int,
        RAW_WORD2ID: word2id,
        RAW_ID2WORD: vocab
    }

    graph_utils.add_dict_to_collection(vocab_lookup, VOCAB_LOOKUP_COLL_NAME)

    return vocab_lookup


def get_vocab_lookup_tables():
    vocab_lookup = graph_utils.get_dict_from_collection(VOCAB_LOOKUP_COLL_NAME)
    return vocab_lookup


def get_token_id(token, vocab_table=None):
    if not vocab_table:
        vocab_table = get_vocab_lookup_tables()[STR_TO_INT]

    token = tf.constant(bytes(token, encoding='utf8'), dtype=tf.string)
    token_id = vocab_table.lookup(token)
    return token_id


def init_embeddings(embed_matrix):
    with tf.variable_scope('embedding_lookup', reuse=False):
        embeddings = tf.get_variable('embeddings',
                                     shape=embed_matrix.shape,
                                     initializer=tf.constant_initializer(embed_matrix),
                                     trainable=True)

    graph_utils.add_dict_to_collection({'matrix': embeddings}, EMBEDDING_MATRIX_COLL_NAME)

    return embeddings


def get_embeddings():
    return graph_utils.get_dict_from_collection(EMBEDDING_MATRIX_COLL_NAME)['matrix']


def embed_tokens(ids):
    embeddings = get_embeddings()
    return tf.nn.embedding_lookup(embeddings, ids)


def get_bpemb_instance(config) -> BPEmb:
    if config.editor.get('use_pretrained_embeddings', True):
        return BPEmb(lang='en', vs=config.editor.vocab_size, dim=config.editor.word_dim, vs_fallback=False)
    else:
        return BPEmb(lang='en', vs=config.editor.vocab_size, vs_fallback=False)


def get_t2t_subword_encoder_instance(config: Config):
    from models.common.subtoken_encoder import SubwordTextEncoder
    vocab_path = str(config.local_data_dir / config.dataset.path / config.editor.t2t_sub_words_vocab_path)
    encoder = SubwordTextEncoder(vocab_path)

    return encoder
