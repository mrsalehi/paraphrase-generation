import random

import tensorflow as tf
from tensorflow.contrib.estimator import InMemoryEvaluatorHook
from tqdm import tqdm

tf.logging.set_verbosity(tf.logging.INFO)

from models.common import vocab, metrics, util
from models.common.sequence import length_pre_embedding
from models.common.vocab import PAD_TOKEN
from models.neural_editor import editor, optimizer, decoder


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
    with open(file_path, encoding='utf8') as f:
        lines = map(lambda x: x[:-1], f)
        examples = map(parse_instance, lines)
        examples = list(tqdm(examples))

        if num_samples and len(examples) > num_samples:
            random.seed(seed)
            examples = random.sample(examples, num_samples)

        return examples


def get_generator(dataset, index):
    def gen():
        for inst in dataset:
            yield inst[index]

    return gen


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
    src_words, tgt_words, inserted_words, deleted_words = features

    vocab_s2i = vocab_tables[vocab.STR_TO_INT]
    vocab_i2s = vocab_tables[vocab.INT_TO_STR]

    train_decoder_output, infer_decoder_output, \
    gold_dec_out, gold_dec_out_len = editor.editor_train(
        src_words, tgt_words, inserted_words, deleted_words, embedding_matrix, vocab_s2i,
        config.editor.hidden_dim, config.editor.agenda_dim, config.editor.edit_dim,
        config.editor.encoder_layers, config.editor.decoder_layers,
        config.editor.attention_dim, config.editor.max_sent_length,
        config.editor.dropout_keep, config.editor.lamb_reg,
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
            evaluation_hooks=[extra_summary_logger]
        )

    elif mode == tf.estimator.ModeKeys.PREDICT:
        preds = {
            'str_tokens': decoder.str_tokens(infer_decoder_output, vocab_i2s),
            'sample_id': decoder.sample_id(infer_decoder_output),
            'lengths': decoder.seq_length(infer_decoder_output),
            'attn_scores': decoder.attention_score(infer_decoder_output)
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


def put_epoch_num(config, data_dir):
    p = data_dir / config.dataset.path / 'train.tsv'
    total_num_examples = util.get_num_total_lines(p)
    num_batch_per_epoch = total_num_examples // config.optim.batch_size
    num_epoch = config.optim.max_iters // num_batch_per_epoch + 1
    config.put('optim.num_epoch', num_epoch)


def train(config, data_dir):
    put_epoch_num(config, data_dir)

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
