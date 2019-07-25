import json
import logging
import pickle
import random

import tensorflow as tf
from tensorflow.python import debug as tf_debug

# import tensorflow.python.util.deprecation as deprecation
#
# deprecation._PRINT_DEPRECATION_WARNINGS = False
from tensorflow.python.util import deprecation

# logging.disable(logging.WARNING)
deprecation._PRINT_DEPRECATION_WARNINGS = False

try:
    from tensorflow.contrib.estimator import InMemoryEvaluatorHook
except:
    from tensorflow.estimator.experimental import InMemoryEvaluatorHook

from tqdm import tqdm

from models.common.util import save_tsv
from models.neural_editor.input import train_input_fn, eval_input_fn, input_fn_cmd, convert_to_bytes, parse_instance, \
    get_generator, input_fn_from_gen_multi
from models.neural_editor.model import model_fn

tf.logging.set_verbosity(tf.logging.INFO)

from models.common import vocab, util
from models.neural_editor import editor, optimizer, decoder, paraphrase_gen

NAME = 'neural_editor'


def get_vocab_embedding_matrix(config, data_dir):
    if config.editor.get('use_sub_words', False):
        if config.editor.get('use_t2t_sub_words'):
            output = vocab.read_t2t_subword_embeddings(config)
        else:
            output = vocab.read_subword_embeddings(config)
    else:
        output = vocab.read_word_embeddings(
            data_dir / 'word_vectors' / config.editor.wvec_path,
            config.editor.word_dim,
            config.editor.vocab_size,
            random_initialization=(not config.editor.get('use_pretrained_embeddings', True)),
            vocab_file=config.editor.get('word_vocab_file_path', None)
        )

    return output


def get_eval_hook(estimator, input_fn, name, every_n_steps):
    return InMemoryEvaluatorHook(
        estimator,
        input_fn,
        name=name,
        every_n_iter=every_n_steps
    )


def get_estimator(config, embed_matrix, my_model_fn=model_fn):
    run_config = tf.estimator.RunConfig(
        model_dir=config.model_dir,
        tf_random_seed=config.seed,
        save_checkpoints_steps=config.eval.save_steps,
        save_summary_steps=config.eval.save_summary_steps,
        keep_checkpoint_max=config.eval.keep_checkpoint_max,
        log_step_count_steps=10
    )

    estimator = tf.estimator.Estimator(
        model_fn=lambda features, labels, mode, params: my_model_fn(
            features,
            mode,
            params,
            embed_matrix,
            vocab.get_vocab_lookup_tables()
        ),
        model_dir=config.model_dir,
        config=run_config,
        params=config
    )

    return estimator


def train(config, data_dir, my_model_fn=model_fn):
    V, embed_matrix = get_vocab_embedding_matrix(config, data_dir)
    estimator = get_estimator(config, embed_matrix, my_model_fn)

    if config.get('eval.enable', True):
        hooks = [
            get_eval_hook(estimator,
                          lambda: eval_input_fn(config, data_dir, vocab.create_vocab_lookup_tables(V)),
                          name='eval',
                          every_n_steps=config.eval.eval_steps),

            # get_eval_hook(estimator,
            #               lambda: eval_input_fn(config, data_dir, vocab.create_vocab_lookup_tables(V),
            #                                     num_examples=config.eval.big_num_examples),
            #               name='eval_big',
            #               every_n_steps=config.eval.big_eval_steps),
            #
            # get_eval_hook(estimator,
            #               lambda: eval_input_fn(config, data_dir, vocab.create_vocab_lookup_tables(V),
            #                                     file_name='train.tsv', num_examples=config.eval.big_num_examples),
            #               name='train_big',
            #               every_n_steps=config.eval.big_eval_steps),
        ]
    else:
        hooks = []

    if config.get('eval.debug.tensorboard', False):
        hooks += [tf_debug.TensorBoardDebugHook('localhost:%s' % config.get('eval.debug.tensorboard_port', 6068),
                                                send_traceback_and_source_code=False)]

    if config.get('eval.debug.cli', False):
        hooks = [tf_debug.LocalCLIDebugHook()]

    return estimator.train(
        input_fn=lambda: train_input_fn(config, data_dir, vocab.create_vocab_lookup_tables(V)),
        hooks=hooks,
        max_steps=config.optim.max_iters
    )


def eval(config, data_dir, checkpoint_path=None, my_model_fn=model_fn):
    V, embed_matrix = vocab.read_word_embeddings(
        data_dir / 'word_vectors' / config.editor.wvec_path,
        config.editor.word_dim,
        config.editor.vocab_size
    )

    estimator = get_estimator(config, embed_matrix, my_model_fn)

    output = estimator.evaluate(
        input_fn=lambda: eval_input_fn(config, data_dir, vocab.create_vocab_lookup_tables(V), num_examples=1e10),
        checkpoint_path=checkpoint_path
    )

    return output


def predict(config, data_dir, checkpoint_path=None, my_model_fn=model_fn):
    V, embed_matrix = vocab.read_word_embeddings(
        data_dir / 'word_vectors' / config.editor.wvec_path,
        config.editor.word_dim,
        config.editor.vocab_size
    )

    config.put('optim.batch_size', 1)

    estimator = get_estimator(config, embed_matrix, my_model_fn)

    output = estimator.predict(
        input_fn=lambda: eval_input_fn(config, data_dir, vocab.create_vocab_lookup_tables(V), num_examples=10),
        checkpoint_path=checkpoint_path
    )

    for p in output:
        print(p['joined'])

    return output


def predict_cmd(config, data_dir, checkpoint_path=None):
    V, embed_matrix = vocab.read_word_embeddings(
        data_dir / 'word_vectors' / config.editor.wvec_path,
        config.editor.word_dim,
        config.editor.vocab_size
    )

    config.put('optim.batch_size', 1)

    estimator = get_estimator(config, embed_matrix)

    output = estimator.predict(
        input_fn=lambda: input_fn_cmd(vocab.create_vocab_lookup_tables(V)),
        checkpoint_path=checkpoint_path
    )

    for p in output:
        print('\nResult:')
        print(p['joined'])

    return output


NUM_CANDIDATES = 5
NUM_SAMPLING = 5


def generate_candidate(train_examples, base):
    candidates = []
    for _ in range(NUM_CANDIDATES):
        possibles = random.sample(train_examples, NUM_SAMPLING)
        candidates.append(
            max(possibles, key=lambda x: x[1])
        )

    return candidates


def augment_dataset(train_examples, estimator, checkpoint_path, classes, V):
    # dtrain, dtest, classes = ds

    index_mapping = []
    augment_formulas = []
    for e, (dtrain, _, _) in enumerate(tqdm(classes)):
        for i, cls in dtrain:
            candidates = generate_candidate(train_examples, i)
            augment_formulas += [(i, c[0], cls) for c in candidates]
            index_mapping += [e] * len(candidates)

    def augment_generator():
        for base, edit, c in augment_formulas:
            base_words = [w.lower() for w in base.split(' ')]
            base_words = tuple([convert_to_bytes(base_words)])
            edit_instance = parse_instance(edit)

            yield base_words + edit_instance

    output = estimator.predict(
        input_fn=lambda: input_fn_from_gen_multi(augment_generator, vocab.create_vocab_lookup_tables(V),
                                                 10),
        checkpoint_path=checkpoint_path
    )

    additional_examples = {}
    for i, p in enumerate(output):
        af = augment_formulas[i]
        episode_id = index_mapping[i]
        if episode_id not in additional_examples:
            additional_examples[episode_id] = []

        ag = p['joined'][0].decode('utf8')
        ag = ' '.join(ag.split(' ')[:-1])

        additional_examples[episode_id].append(
            (ag, af[2])
        )
        try:
            print("cls:\t", af[2])
            print("base:\t", af[0])
            edit = af[1]
            src, tgt, iw, dw = parse_instance(edit)
            print("src:\t", src)
            print("iw:\t", iw)
            print("dw:\t", dw)
            print("tgt:\t", tgt)
            print('augmented:\t', ag)
        except Exception as e:
            print(e)
            pass
        print("###")
        print(i + 1, len(augment_formulas))
        print("===============================================\n\n")

    # dtrain += additional_examples

    for episode_id in additional_examples:
        v = classes[episode_id][0] + additional_examples[episode_id]
        assert len(v) == len(classes[episode_id][0]) * (1 + NUM_CANDIDATES)
        for _ in range(3):
            random.shuffle(v)

        additional_examples[episode_id] = v

    return additional_examples


def augment_meta_test(config, meta_test_path, data_dir, checkpoint_path=None, my_model_fn=model_fn):
    V, embed_matrix = vocab.read_word_embeddings(
        data_dir / 'word_vectors' / config.editor.wvec_path,
        config.editor.word_dim,
        config.editor.vocab_size
    )

    config.put('optim.batch_size', 1)

    estimator = get_estimator(config, embed_matrix, my_model_fn)

    with open(str(data_dir / config.dataset.path / 'train.tsv'), encoding='utf8') as f:
        train_examples = []
        for l in tqdm(f, total=util.get_num_total_lines(data_dir / config.dataset.path / 'train.tsv')):
            l = l[:-1]
            src, tgt = l.split('\t')

            train_examples.append((l, util.jaccard(
                set([w.lower() for w in src.split(' ')]),
                set([w.lower() for w in tgt.split(' ')]),
            )))

        train_examples = list(filter(lambda x: 0.6 < x[1] < 0.8, train_examples))

    with open(meta_test_path, 'rb') as f:
        meta_test = pickle.load(f)

    # for i, m in enumerate(tqdm(meta_test)):
    #     dtrain = augment_dataset(train_examples, estimator, checkpoint_path, meta_test[0], V)
    #     assert len(dtrain) == len(meta_test[0]) * (1 + NUM_CANDIDATES)
    #     print("DTRRAAAAAAAAAAAAAIIIIIIIIIIIINNNNNNNN:", len(dtrain))
    #     meta_test[i] = (dtrain, meta_test[i][1], meta_test[i][2])

    new_meta_test = []
    all_dtrain = augment_dataset(train_examples, estimator, checkpoint_path, meta_test, V)
    for episode_id in all_dtrain:
        new_meta_test.append((
            all_dtrain[episode_id],
            meta_test[episode_id][1],
            meta_test[episode_id][2],
        ))

    with open(meta_test_path + '_augmented.pkl', 'wb') as f:
        pickle.dump(new_meta_test, f)


def augment_debug_dataset(debug_examples, estimator, checkpoint_path, V):
    input_examples = []
    mapping = {}
    for i, (base, src, dst, c) in enumerate(debug_examples):
        mapping[i] = (base, src, dst, c)
        input_examples.append((base, '\t'.join([src, dst])))

    def augment_generator():
        for base, edit in input_examples:
            base_words = [w.lower() for w in base.split(' ')]
            base_words = tuple([convert_to_bytes(base_words)])
            edit_instance = parse_instance(edit)

            yield base_words + edit_instance

    output = estimator.predict(
        input_fn=lambda: input_fn_from_gen_multi(augment_generator, vocab.create_vocab_lookup_tables(V), 10),
        checkpoint_path=checkpoint_path
    )

    result = []
    for i, p in enumerate(output):
        af = mapping[i]

        ag = p['joined'][0].decode('utf8')
        ag = ' '.join(ag.split(' ')[:-1])

        result.append(af + (ag,))

    return result


def augment_debug(config, debug_dataset, data_dir, checkpoint_path=None, my_model_fn=model_fn):
    V, embed_matrix = vocab.read_word_embeddings(
        data_dir / 'word_vectors' / config.editor.wvec_path,
        config.editor.word_dim,
        config.editor.vocab_size
    )

    config.put('optim.batch_size', 1)

    estimator = get_estimator(config, embed_matrix, my_model_fn)

    with open(str(data_dir / debug_dataset), 'rb') as f:
        debug_examples = pickle.load(f)

    debugged = augment_debug_dataset(debug_examples, estimator, checkpoint_path, V)

    with open("%s_debugged" % debug_dataset, 'w', encoding='utf8') as f:
        json.dump(debugged, f)


def generate_paraphrase(config, data_dir, checkpoint_path, plan_path, output_path, beam_width, batch_size,
                        my_model_fn=model_fn):
    V, embed_matrix = get_vocab_embedding_matrix(config, data_dir)

    if batch_size:
        config.put('optim.batch_size', batch_size)

    if beam_width:
        config.put('editor.beam_width', beam_width)

    estimator = get_estimator(config, embed_matrix, my_model_fn)

    paras, attn_weights, base_attn_weight, mev_attn_weight = paraphrase_gen.generate(estimator, plan_path,
                                                                                     checkpoint_path, config, V)
    flatten = paraphrase_gen.flatten(paras)

    save_tsv(output_path, flatten)
    paraphrase_gen.save_attn_weights(attn_weights, '%s.attn_weights' % output_path)
    paraphrase_gen.save_attn_weights(base_attn_weight, '%s.base_attn_weights' % output_path)
    paraphrase_gen.save_attn_weights(mev_attn_weight, '%s.mev_attn_weights' % output_path)
