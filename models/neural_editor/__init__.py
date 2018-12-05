import pickle
import random

import tensorflow as tf
import tensorflow.contrib.metrics as tf_metrics
from tensorflow.contrib.estimator import InMemoryEvaluatorHook
from tqdm import tqdm

tf.logging.set_verbosity(tf.logging.INFO)

from models.common import vocab, metrics, util
from models.common.sequence import length_pre_embedding
from models.common.vocab import PAD_TOKEN
from models.neural_editor import editor, optimizer, decoder

DATASET_CACHE = {}


def convert_to_bytes(lst):
    return [bytes(w, encoding='utf8') for w in lst]


def parse_instance(instance_str):
    src, tgt = instance_str.split('\t')

    src_words = [w.lower() for w in src.split(' ')]
    tgt_words = [w.lower() for w in tgt.split(' ')]

    insert_words = sorted(set(tgt_words) - set(src_words))
    delete_words = sorted(set(src_words) - set(tgt_words))

    if len(insert_words) == 0:
        insert_words.append(vocab.UNKNOWN_TOKEN)

    if len(delete_words) == 0:
        delete_words.append(vocab.UNKNOWN_TOKEN)

    return convert_to_bytes(src_words), \
           convert_to_bytes(tgt_words), \
           convert_to_bytes(insert_words), \
           convert_to_bytes(delete_words)


def read_examples_from_file(file_path, num_samples=None, seed=0):
    if not isinstance(file_path, str):
        file_path = str(file_path)

    if file_path in DATASET_CACHE:
        print('Reading examples from cache...')
        examples = DATASET_CACHE[file_path]
    else:
        print('Reading examples from %s...' % file_path)
        with open(file_path, encoding='utf8') as f:
            lines = map(lambda x: x[:-1], f)
            examples = map(parse_instance, lines)
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


def input_fn(file_path, vocab_table, batch_size, num_epochs=None, num_examples=None, seed=0):
    if isinstance(vocab_table, dict):
        vocab_table = vocab_table[vocab.STR_TO_INT]

    pad_token = tf.constant(bytes(PAD_TOKEN, encoding='utf8'), dtype=tf.string)
    pad_value = vocab_table.lookup(pad_token)

    base_dataset = read_examples_from_file(file_path, num_examples, seed)

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
        config.seed
    )


def eval_input_fn(config, data_dir, vocab_table, file_name='valid.tsv', num_examples=None):
    if not num_examples:
        num_examples = config.eval.num_examples

    return input_fn(
        data_dir / config.dataset.path / file_name,
        vocab_table,
        config.optim.batch_size,
        num_examples=num_examples,
        seed=config.seed
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


def get_train_extra_summary_writer(model_dir, save_steps):
    return tf.train.SummarySaverHook(
        save_steps=save_steps,
        output_dir=model_dir,
        summary_op=tf.summary.merge_all('extra')
    )


def get_extra_summary_logger(pred_op, bleu_op, every_n_step):
    tensors = {
        'pred': pred_op,
        'bleu': bleu_op
    }

    def print_pred_summary(pred_result):
        output_str = ''
        for src, iw, dw, tgt, pred in pred_result:
            output_str += 'SOURCE: %s\n' % src
            output_str += 'INSERT: %s\n' % iw
            output_str += 'DELETE: %s\n' % dw
            output_str += 'TARGET: %s\n\n' % tgt
            output_str += '%s\n' % pred
            output_str += '####################################################################\n'

        return output_str

    def print_bleu_score(bleu_result):
        return 'BLEU: %s' % bleu_result

    return tf.train.LoggingTensorHook(
        tensors=tensors,
        every_n_iter=every_n_step,
        formatter=lambda t: '\n'.join([print_pred_summary(t['pred']), print_bleu_score(t['bleu'])])
    )


def get_eval_hook(estimator, input_fn, name, every_n_steps):
    return InMemoryEvaluatorHook(
        estimator,
        input_fn,
        name=name,
        every_n_iter=every_n_steps
    )


def add_extra_summary(vocab_i2s, decoder_output, src_words, tgt_words, inserted_words, deleted_words, collections=None):
    pred_tokens = decoder.str_tokens(decoder_output, vocab_i2s)
    pred_len = decoder.seq_length(decoder_output)

    tgt_tokens = vocab_i2s.lookup(tgt_words)
    tgt_len = length_pre_embedding(tgt_words)

    bleu_score = metrics.create_bleu_metric_ops(tgt_tokens, pred_tokens, tgt_len, pred_len)
    avg_bleu = tf.reduce_mean(bleu_score)
    tf.summary.scalar('bleu', avg_bleu, collections)

    pred_joined = metrics.join_tokens(pred_tokens, pred_len)
    tgt_joined = metrics.join_tokens(tgt_tokens, tgt_len)
    src_joined = metrics.join_tokens(vocab_i2s.lookup(src_words), length_pre_embedding(src_words))
    iw_joined = metrics.join_tokens(vocab_i2s.lookup(inserted_words), length_pre_embedding(inserted_words), ', ')
    dw_joined = metrics.join_tokens(vocab_i2s.lookup(deleted_words), length_pre_embedding(deleted_words), ', ')

    pred_summary = tf.concat([src_joined, iw_joined, dw_joined, tgt_joined, pred_joined], axis=1)
    tf.summary.text('trace', pred_summary, collections)

    return avg_bleu, pred_summary


def model_fn(features, mode, config, embedding_matrix, vocab_tables):
    if mode == tf.estimator.ModeKeys.PREDICT:
        base_words, src_words, tgt_words, inserted_words, deleted_words = features
    else:
        src_words, tgt_words, inserted_words, deleted_words = features
        base_words = src_words

    vocab_s2i = vocab_tables[vocab.STR_TO_INT]
    vocab_i2s = vocab_tables[vocab.INT_TO_STR]

    train_decoder_output, infer_decoder_output, \
    gold_dec_out, gold_dec_out_len = editor.editor_train(
        base_words, src_words, tgt_words, inserted_words, deleted_words, embedding_matrix, vocab_s2i,
        config.editor.hidden_dim, config.editor.agenda_dim, config.editor.edit_dim,
        config.editor.encoder_layers, config.editor.decoder_layers, config.editor.attention_dim,
        config.editor.edit_enc.ctx_hidden_dim, config.editor.edit_enc.ctx_hidden_layer,
        config.editor.edit_enc.wa_hidden_dim, config.editor.edit_enc.wa_hidden_layer,
        config.editor.max_sent_length, config.editor.dropout_keep, config.editor.lamb_reg,
        config.editor.norm_eps, config.editor.norm_max, config.editor.kill_edit,
        config.editor.draw_edit, config.editor.use_swap_memory
    )

    loss = optimizer.loss(train_decoder_output, gold_dec_out, gold_dec_out_len)
    train_op, gradients_norm = optimizer.train(loss, config.optim.learning_rate, config.optim.max_norm_observe_steps)

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('grad_norm', gradients_norm)
        avg_bleu, pred_summary = add_extra_summary(vocab_i2s, train_decoder_output,
                                                   src_words, tgt_words, inserted_words, deleted_words,
                                                   ['extra'])

        extra_summary_logger = get_extra_summary_logger(pred_summary, avg_bleu, config.eval.eval_steps)
        extra_summary_writer = get_train_extra_summary_writer(config.model_dir, config.eval.eval_steps)

        return tf.estimator.EstimatorSpec(
            mode,
            train_op=train_op,
            loss=loss,
            training_hooks=[extra_summary_writer, extra_summary_logger]
        )

    elif mode == tf.estimator.ModeKeys.EVAL:
        avg_bleu, pred_summary = add_extra_summary(vocab_i2s, train_decoder_output,
                                                   src_words, tgt_words, inserted_words, deleted_words)

        extra_summary_logger = get_extra_summary_logger(pred_summary, avg_bleu, 1)
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            evaluation_hooks=[extra_summary_logger],
            eval_metric_ops={'bleu': tf_metrics.streaming_mean(avg_bleu)}
        )

    elif mode == tf.estimator.ModeKeys.PREDICT:
        # attn_scores = decoder.attention_score(infer_decoder_output)
        lengths = decoder.seq_length(infer_decoder_output)
        tokens = decoder.str_tokens(infer_decoder_output, vocab_i2s)
        preds = {
            'str_tokens': tokens,
            'sample_id': decoder.sample_id(infer_decoder_output),
            'lengths': lengths,
            'joined': metrics.join_tokens(tokens, lengths),
            # 'attn_scores_0': attn_scores[0],
            # 'attn_scores_1': attn_scores[1],
            # 'attn_scores_2': attn_scores[2],
        }
        return tf.estimator.EstimatorSpec(
            mode,
            predictions=preds
        )


def get_estimator(config, embed_matrix):
    run_config = tf.estimator.RunConfig(
        model_dir=config.model_dir,
        tf_random_seed=config.seed,
        save_checkpoints_steps=config.eval.save_steps,
        save_summary_steps=config.eval.save_summary_steps,
        keep_checkpoint_max=config.eval.keep_checkpoint_max,
        log_step_count_steps=1
    )

    estimator = tf.estimator.Estimator(
        model_fn=lambda features, labels, mode, params: model_fn(
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


def train(config, data_dir):
    V, embed_matrix = vocab.read_word_embeddings(
        data_dir / 'word_vectors' / config.editor.wvec_path,
        config.editor.word_dim,
        config.editor.vocab_size
    )

    estimator = get_estimator(config, embed_matrix)

    return estimator.train(
        input_fn=lambda: train_input_fn(config, data_dir, vocab.create_vocab_lookup_tables(V)),
        hooks=[
            get_eval_hook(estimator,
                          lambda: eval_input_fn(config, data_dir, vocab.create_vocab_lookup_tables(V)),
                          name='eval',
                          every_n_steps=config.eval.eval_steps),

            get_eval_hook(estimator,
                          lambda: eval_input_fn(config, data_dir, vocab.create_vocab_lookup_tables(V),
                                                num_examples=config.eval.big_num_examples),
                          name='eval_big',
                          every_n_steps=config.eval.big_eval_steps),

            get_eval_hook(estimator,
                          lambda: eval_input_fn(config, data_dir, vocab.create_vocab_lookup_tables(V),
                                                file_name='train.tsv', num_examples=config.eval.big_num_examples),
                          name='train_big',
                          every_n_steps=config.eval.big_eval_steps),
        ]
    )


def eval(config, data_dir, checkpoint_path=None):
    V, embed_matrix = vocab.read_word_embeddings(
        data_dir / 'word_vectors' / config.editor.wvec_path,
        config.editor.word_dim,
        config.editor.vocab_size
    )

    estimator = get_estimator(config, embed_matrix)

    output = estimator.evaluate(
        input_fn=lambda: eval_input_fn(config, data_dir, vocab.create_vocab_lookup_tables(V), num_examples=1e10),
        checkpoint_path=checkpoint_path
    )

    return output


def predict(config, data_dir, checkpoint_path=None):
    V, embed_matrix = vocab.read_word_embeddings(
        data_dir / 'word_vectors' / config.editor.wvec_path,
        config.editor.word_dim,
        config.editor.vocab_size
    )

    config.put('optim.batch_size', 1)

    estimator = get_estimator(config, embed_matrix)

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


def augment_dataset(train_examples, estimator, checkpoint_path, ds, V):
    dtrain, dtest, classes = ds

    augment_formulas = []
    for i, cls in dtrain:
        candidates = generate_candidate(train_examples, i)
        augment_formulas += [(i, c[0], cls) for c in candidates]

    def augment_generator():
        for base, edit, c in augment_formulas:
            base_words = [w.lower() for w in base.split(' ')]
            base_words = tuple([convert_to_bytes(base_words)])
            edit_instance = parse_instance(edit)

            yield base_words + edit_instance

    output = estimator.predict(
        input_fn=lambda: input_fn_from_gen(augment_generator, vocab.create_vocab_lookup_tables(V)),
        checkpoint_path=checkpoint_path
    )

    additional_examples = []
    for i, p in enumerate(output):
        af = augment_formulas[i]
        print("cls:\t", af[2])
        print("base:\t", af[0])
        edit = af[1]
        src, tgt, iw, dw = parse_instance(edit)
        print("src:\t", src)
        print("iw:\t", iw)
        print("dw:\t", dw)
        print("tgt:\t", tgt)
        print('augmented:\t', p['joined'][0])
        additional_examples.append(
            (p['joined'][0].decode('utf8'), af[2])
        )
        print("===============================================\n\n")

    dtrain += additional_examples

    for _ in range(3):
        random.shuffle(dtrain)

    return dtrain


def augment_meta_test(config, meta_test_path, data_dir, checkpoint_path=None):
    V, embed_matrix = vocab.read_word_embeddings(
        data_dir / 'word_vectors' / config.editor.wvec_path,
        config.editor.word_dim,
        config.editor.vocab_size
    )

    config.put('optim.batch_size', 1)

    estimator = get_estimator(config, embed_matrix)

    with open(data_dir / config.dataset.path / 'train.tsv', encoding='utf8') as f:
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

    for i, m in enumerate(tqdm(meta_test)):
        dtrain = augment_dataset(train_examples, estimator, checkpoint_path, meta_test[0], V)
        print(len(data_dir))
        meta_test[i][0] = dtrain

    with open(meta_test_path+'_augmented.pkl', 'rb') as f:
        pickle.dump(meta_test, f)
