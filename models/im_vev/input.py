from models.neural_editor import convert_to_bytes

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
    base_words = base.lower().split(' ')
    output_words = output.lower().split(' ')

    src, tgt = instance[2:]
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

    return convert_to_bytes(base_words), \
           convert_to_bytes(output_words), \
           convert_to_bytes(src_words), \
           convert_to_bytes(tgt_words), \
           convert_to_bytes(insert_words), \
           convert_to_bytes(delete_words)


def read_examples_from_file(file_path, num_samples=None, seed=0, noiser=None, free_set=None):
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
            examples = list(tqdm(examples, total=util.get_num_total_lines(file_path)))

        DATASET_CACHE[file_path] = examples

    if num_samples and len(examples) > num_samples:
        random.seed(seed)
        examples = random.sample(examples, num_samples)

    return examples