from models.neural_editor.edit_noiser import EditNoiser

DATASET_CACHE = {}

import random

import tensorflow as tf
from tqdm import tqdm

tf.logging.set_verbosity(tf.logging.INFO)

from models.common import vocab, util
from models.common.vocab import PAD_TOKEN


def convert_to_bytes(lst):
    return [bytes(w, encoding='utf8') for w in lst]


def parse_instance(instance, noiser=None, free=None):
    if isinstance(instance, str):
        instance = instance.split('\t')

    src, tgt = instance

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

    return convert_to_bytes(src_words), \
           convert_to_bytes(tgt_words), \
           convert_to_bytes(insert_words), \
           convert_to_bytes(delete_words)


def read_examples_from_file(file_path, num_samples=None, seed=0, noiser=None, free_set=None):
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


def get_generator(dataset, index):
    def gen():
        for inst in dataset:
            yield inst[index]

    return gen


def input_fn_cmd(vocab_table):
    def get_single_cmd_input():
        base = input("Enter Base:\n")
        src = input("Enter Source:\n")
        tgt = input("Enter Target:\n")
        insert_words = input("Enter Insert words:\n")
        delete_words = input("Enter Delete words:\n")

        base_words = [w.lower() for w in base.split(' ')]
        base_words = [convert_to_bytes(base_words)]

        if insert_words == '@' and delete_words == '@':
            edit_instance = parse_instance('%s\t\%s' % (src, tgt))
            edit_instance = tuple([tuple(i) for i in edit_instance])
        else:
            src_words = [w.lower() for w in src.split(' ')]
            tgt_words = [w.lower() for w in tgt.split(' ')]
            insert_words = [w.lower() for w in insert_words.split(' ')]
            delete_words = [w.lower() for w in delete_words.split(' ')]
            if len(insert_words) == 0:
                insert_words.append(vocab.UNKNOWN_TOKEN)
            if len(delete_words) == 0:
                delete_words.append(vocab.UNKNOWN_TOKEN)

            edit_instance = tuple(convert_to_bytes(src_words)), \
                            tuple(convert_to_bytes(tgt_words)), \
                            tuple(convert_to_bytes(insert_words)), \
                            tuple(convert_to_bytes(delete_words))

        return tuple(base_words) + edit_instance

    def dataset_generator():
        while True:
            print("\n\n\n")
            yield get_single_cmd_input()

    if isinstance(vocab_table, dict):
        vocab_table = vocab_table[vocab.STR_TO_INT]

    dataset = tf.data.Dataset.from_generator(
        generator=dataset_generator,
        output_types=(tf.string, tf.string, tf.string, tf.string, tf.string),
        output_shapes=((None,), (None,), (None,), (None,), (None,))
    )
    dataset = dataset.map(lambda *x: [vocab_table.lookup(i) for i in x])
    dataset = dataset.batch(1)

    fake_label = tf.data.Dataset.from_tensor_slices(tf.constant([0])).repeat()

    dataset = dataset.zip((dataset, fake_label))

    return dataset


def input_fn_from_gen(gen, vocab_table):
    if isinstance(vocab_table, dict):
        vocab_table = vocab_table[vocab.STR_TO_INT]

    dataset = tf.data.Dataset.from_generator(
        generator=gen,
        output_types=(tf.string, tf.string, tf.string, tf.string, tf.string),
        output_shapes=((None,), (None,), (None,), (None,), (None,))
    )
    dataset = dataset.map(lambda *x: [vocab_table.lookup(i) for i in x])
    dataset = dataset.batch(1)

    fake_label = tf.data.Dataset.from_tensor_slices(tf.constant([0])).repeat()

    dataset = dataset.zip((dataset, fake_label))

    return dataset


def input_fn_from_gen_multi(gen, vocab_table, batch_size):
    if isinstance(vocab_table, dict):
        vocab_table = vocab_table[vocab.STR_TO_INT]

    base_dataset = list(gen())

    pad_token = tf.constant(bytes(PAD_TOKEN, encoding='utf8'), dtype=tf.string)
    pad_value = vocab_table.lookup(pad_token)

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

    dataset = tf.data.Dataset.zip(tuple(dataset_splits))

    fake_label = tf.data.Dataset.from_tensor_slices(tf.constant([0])).repeat()

    dataset = dataset.zip((dataset, fake_label))

    return dataset


def input_fn(file_path, vocab_table, batch_size, num_epochs=None, num_examples=None, seed=0, noiser=None,
             use_free_set=False):
    if isinstance(vocab_table, dict):
        vocab_table = vocab_table[vocab.STR_TO_INT]

    pad_token = tf.constant(bytes(PAD_TOKEN, encoding='utf8'), dtype=tf.string)
    pad_value = vocab_table.lookup(pad_token)

    base_dataset = read_examples_from_file(
        file_path, num_examples, seed,
        noiser, util.get_free_words_set() if use_free_set else None
    )

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

    dataset = tf.data.Dataset.zip(tuple(dataset_splits))
    if num_epochs:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000, num_epochs))

    fake_label = tf.data.Dataset.from_tensor_slices(tf.constant([0])).repeat()

    dataset = dataset.zip((dataset, fake_label)) \
        .prefetch(1)

    return dataset


def train_input_fn(config, data_dir, vocab_table):
    return input_fn(
        data_dir / config.dataset.path / 'train.tsv',
        vocab_table,
        config.optim.batch_size,
        config.optim.num_epoch,
        config.seed,
        noiser=EditNoiser.from_config(config),
        use_free_set=config.editor.use_free_set
    )


def eval_input_fn(config, data_dir, vocab_table, file_name='valid.tsv', num_examples=None):
    if not num_examples:
        num_examples = config.eval.num_examples

    return input_fn(
        data_dir / config.dataset.path / file_name,
        vocab_table,
        config.optim.batch_size,
        num_examples=num_examples,
        seed=config.seed,
        noiser=EditNoiser.from_config(config),
        use_free_set=config.editor.use_free_set
    )


def eval_big_input_fn(config, data_dir, vocab_table):
    return eval_input_fn(
        config,
        data_dir,
        vocab_table,
        num_examples=config.eval.big_num_examples
    )


def train_big_input_fn(config, data_dir, vocab_table):
    return eval_input_fn(
        config,
        data_dir,
        vocab_table,
        file_name='train.tsv',
        num_examples=config.eval.big_num_examples
    )
