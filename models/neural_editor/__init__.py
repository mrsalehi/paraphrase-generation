import tensorflow as tf

from models.common.vocab import get_vocab_lookup, read_word_embeddings, PAD_TOKEN


def convert_to_bytes(lst):
    return [bytes(w, encoding='utf8') for w in lst]


def parse_instance(instance_str):
    src, tgt = instance_str.split('\t')

    src_words = src.split(' ')
    tgt_words = tgt.split(' ')

    insert_words = sorted(set(tgt_words) - set(src_words))
    delete_words = sorted(set(src_words) - set(tgt_words))

    return convert_to_bytes(src_words), \
           convert_to_bytes(tgt_words), \
           convert_to_bytes(insert_words), \
           convert_to_bytes(delete_words)


def read_examples_from_file(file_path):
    with open(file_path, encoding='utf8') as f:
        lines = map(lambda x: x[:-1], f)
        examples = map(parse_instance, lines)
        return list(examples)


def get_generator(dataset, index):
    def gen():
        for inst in dataset:
            yield inst[index]

    return gen


def input_fn(file_path, vocab_table, batch_size, num_epochs):
    pad_token = tf.constant(bytes(PAD_TOKEN, encoding='utf8'), dtype=tf.string)
    pad_value = vocab_table.lookup(pad_token)

    base_dataset = read_examples_from_file(file_path)

    dataset_splits = []
    for index in range(len(base_dataset[0])):
        split = tf.data.Dataset.from_generator(
            generator=get_generator(base_dataset, index),
            output_types=(tf.string),
            output_shapes=((None,))
        )
        split = split.map(lambda x: vocab_table.lookup(x))
        split = split.padded_batch(
            batch_size,
            padded_shapes=[None],
            padding_values=(pad_value)
        )

        dataset_splits.append(split)

    dataset = tf.data.Dataset.zip(tuple(dataset_splits)) \
        .apply(tf.contrib.data.shuffle_and_repeat(1000, num_epochs))

    fake_label_dataset = tf.data.Dataset.from_tensor_slices(tf.constant([0])).repeat()

    dataset = dataset.zip((dataset, fake_label_dataset)) \
        .prefetch(1)

    return dataset


def train_input_fn():
    return input_fn('')


def eval_input_fn():
    return input_fn('')


def get_estimator():
    pass
