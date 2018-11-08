import tensorflow.contrib.lookup as lookup
import numpy as np

SPECIAL_TOKENS = [
    '<unk>',
    '<start>',
    '<stop>'
]

OOV_TOKEN_ID = 0


def emulate_distribution(shape, target_samples):
    m = np.mean(target_samples)
    s = np.std(target_samples)
    samples = np.random.normal(m, s, size=shape)

    return samples


def get_special_tokens_embeds(embeddings):
    shape = (len(SPECIAL_TOKENS), embeddings.shape[1])
    special_embeddings = emulate_distribution(shape, embeddings)
    special_embeddings = special_embeddings.astype(np.float32)

    return special_embeddings


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
        special_tokens_embeddings = get_special_tokens_embeds(embedding_matrix)
        embedding_matrix = np.concatenate([special_tokens_embeddings, embedding_matrix])
        assert embedding_matrix.shape == (vocab_size + len(SPECIAL_TOKENS), embed_dim)

    return vocab, embedding_matrix


def get_vocab_lookup(vocab, name='vocab_lookup'):
    vocab_lookup = lookup.index_table_from_tensor(
        mapping=vocab,
        num_oov_buckets=0,
        default_value=OOV_TOKEN_ID,
        name=name
    )

    return vocab_lookup
