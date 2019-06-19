from bpemb import BPEmb

from models.common.vocab import PAD_TOKEN
from models.neural_editor.edit_noiser import EditNoiser
from models.neural_editor.input import convert_to_bytes, get_generator

DATASET_CACHE = {}

import random

import tensorflow as tf
from tqdm import tqdm

tf.logging.set_verbosity(tf.logging.INFO)

from models.common import vocab, util


def parse_instance(instance, noiser=None, free=None):
    if isinstance(instance, str):
        instance = instance.split('\t')

    assert len(instance) == 4

    base, output = instance[:2]

    src, tgt = instance[2:]
    src_words = src.lower().split(' ')
    tgt_words = tgt.lower().split(' ')

    if free is None:
        free = set()

    insert_words = sorted(set(tgt_words) - set(src_words) - free)
    delete_words = sorted((set(src_words) & set(tgt_words)) - free)

    if noiser:
        src_words, tgt_words, insert_words, delete_words = noiser(
            (src_words, tgt_words, insert_words, delete_words)
        )

    if len(insert_words) == 0:
        insert_words.append(vocab.UNKNOWN_TOKEN)

    if len(delete_words) == 0:
        delete_words.append(vocab.UNKNOWN_TOKEN)

    return base, \
           output, \
           src, \
           tgt, \
           ' '.join(insert_words), \
           ' '.join(delete_words)


def map_word_to_sub_words(instance, bpemb: BPEmb):
    return tuple(bpemb.encode(instance))


def map_str_to_bytes(instance):
    if isinstance(instance, tuple):
        return tuple([convert_to_bytes(s) for s in instance])
    else:
        return convert_to_bytes(instance)


def read_examples_from_file(file_path, config, num_samples=None, seed=0, noiser=None, free_set=None):
    print("new input")
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

            if config.editor.get('use_sub_words', False):
                bpemb = vocab.get_bpemb_instance(config)
                examples = map(lambda x: map_word_to_sub_words(x, bpemb), examples)

            examples = map(map_str_to_bytes, examples)
            examples = list(tqdm(examples, total=util.get_num_total_lines(file_path)))

        DATASET_CACHE[file_path] = examples

    if num_samples and len(examples) > num_samples:
        random.seed(seed)
        examples = random.sample(examples, num_samples)

    return examples


def input_fn(file_path, vocab_table, config, batch_size, num_epochs=None, num_examples=None, seed=0, noiser=None,
             use_free_set=False, shuffle_input=True):
    base_dataset = read_examples_from_file(
        file_path, config, num_examples, seed,
        noiser, util.get_free_words_set() if use_free_set else None
    )

    gen = lambda: iter(base_dataset)

    return input_fn_from_gen_multi(
        gen,
        vocab_table, batch_size,
        shuffle_input=shuffle_input,
        num_epochs=num_epochs,
        prefetch=True
    )


def input_fn_from_gen_multi(gen, vocab_table, batch_size, shuffle_input=False, num_epochs=None, prefetch=False):
    if isinstance(vocab_table, dict):
        vocab_table = vocab_table[vocab.STR_TO_INT]

    base_dataset = list(gen())

    pad_token = tf.constant(bytes(PAD_TOKEN, encoding='utf8'), dtype=tf.string)
    pad_id = vocab_table.lookup(pad_token)

    dataset_splits = []
    for index in range(len(base_dataset[0])):
        split = tf.data.Dataset.from_generator(
            generator=get_generator(base_dataset, index),
            output_types=(tf.string),
            output_shapes=(None,)
        )
        split = split.map(lambda x: vocab_table.lookup(x))
        split = split.padded_batch(
            batch_size,
            padded_shapes=[None],
            padding_values=(pad_id)
        )

        dataset_splits.append(split)

    dataset = tf.data.Dataset.zip(tuple(dataset_splits))
    if num_epochs and shuffle_input:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(500, num_epochs))
    elif num_epochs:
        dataset = dataset.repeat(num_epochs)

    fake_label = tf.data.Dataset.from_tensor_slices(tf.constant([0])).repeat()

    dataset = dataset.zip((dataset, fake_label))
    if prefetch:
        dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return dataset


def train_input_fn(config, data_dir, vocab_table):
    return input_fn(
        data_dir / config.dataset.path / 'train.tsv',
        vocab_table,
        config,
        config.optim.batch_size,
        num_epochs=config.optim.num_epoch,
        seed=config.seed,
        noiser=EditNoiser.from_config(config),
        use_free_set=config.editor.use_free_set,
        shuffle_input=config.get('optim.shuffle_input', False)
    )


def eval_input_fn(config, data_dir, vocab_table, file_name='valid.tsv', num_examples=None):
    if not num_examples:
        num_examples = config.eval.num_examples

    return input_fn(
        data_dir / config.dataset.path / file_name,
        vocab_table,
        config,
        config.optim.batch_size,
        num_examples=num_examples,
        seed=config.seed,
        noiser=EditNoiser.from_config(config),
        use_free_set=config.editor.use_free_set,
        shuffle_input=config.get('optim.shuffle_input', False)
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
