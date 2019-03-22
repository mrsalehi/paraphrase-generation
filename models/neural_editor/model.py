import os

import tensorflow as tf
import tensorflow.contrib.metrics as tf_metrics
from tensorflow.contrib.hooks import ProfilerHook

tf.logging.set_verbosity(tf.logging.INFO)

from models.common import vocab, metrics
from models.common.sequence import length_pre_embedding
from models.neural_editor import editor, optimizer, decoder

ES_TRACE = 'trace'
ES_BLEU = 'bleu'


def get_train_extra_summary_writer(config):
    return tf.train.SummarySaverHook(
        save_steps=config.eval.eval_steps,
        output_dir=config.model_dir,
        summary_op=tf.summary.merge_all('extra')
    )


def get_profiler_hook(config):
    save_steps = config.get('logger.profile_steps', config.eval.eval_steps)
    output_dir = os.path.join(config.model_dir, 'device_profiler')

    return ProfilerHook(
        save_steps=save_steps,
        output_dir=output_dir,
        show_dataflow=True,
        show_memory=True
    )


def get_extra_summary_logger(ops, config):
    formatters = {}

    def print_pred_summary(pred_result):
        output_str = ''
        for src, iw, dw, tgt, pred in pred_result:
            output_str += 'SOURCE: %s\n' % str(src, encoding='utf8')
            output_str += 'INSERT: [ %s ]\n' % str(iw, encoding='utf8')
            output_str += 'DELETE: [ %s ]\n' % str(dw, encoding='utf8')
            output_str += 'TARGET: %s\n' % str(tgt, encoding='utf8')
            output_str += '---\n%s\n---\n\n' % str(pred, encoding='utf8')
            output_str += '=========================\n\n'

        return output_str

    formatters[ES_TRACE] = print_pred_summary

    def print_bleu_score(bleu_result):
        return 'BLEU: %s' % bleu_result

    formatters[ES_BLEU] = print_bleu_score

    return tf.train.LoggingTensorHook(
        tensors=ops,
        every_n_iter=config.eval.eval_steps,
        formatter=lambda x: '\n'.join([formatters[name](t) for name, t in x.items()])
    )


def get_avg_bleu_smmary(tgt_tokens, pred_tokens, tgt_len, pred_len):
    if pred_tokens.shape.ndims > 2:
        best_tokens = pred_tokens[:, :, 0]
        best_len = pred_len[:, 0]
    else:
        best_tokens = pred_tokens
        best_len = pred_len

    bleu_score = metrics.create_bleu_metric_ops(tgt_tokens, best_tokens, tgt_len, best_len)
    avg_bleu = tf.reduce_mean(bleu_score)

    return avg_bleu


def get_trace_summary(vocab_i2s,
                      pred_tokens, tgt_tokens,
                      src_words, inserted_words, deleted_words,
                      pred_len, tgt_len):
    if pred_tokens.shape.ndims > 2:
        pred_joined = metrics.join_beams(pred_tokens, pred_len)
    else:
        pred_joined = metrics.join_tokens(pred_tokens, pred_len)

    tgt_joined = metrics.join_tokens(tgt_tokens, tgt_len)
    src_joined = metrics.join_tokens(vocab_i2s.lookup(src_words), length_pre_embedding(src_words))
    iw_joined = metrics.join_tokens(vocab_i2s.lookup(inserted_words), length_pre_embedding(inserted_words), ', ')
    dw_joined = metrics.join_tokens(vocab_i2s.lookup(deleted_words), length_pre_embedding(deleted_words), ', ')

    return tf.concat([src_joined, iw_joined, dw_joined, tgt_joined, pred_joined], axis=1)


def add_extra_summary(config, vocab_i2s, decoder_output, src_words, tgt_words, inserted_words, deleted_words,
                      collections=None):
    ops = {}

    if config.get('logger.enable_trace', False):
        trace_summary = add_extra_summary_trace(vocab_i2s,
                                                decoder_output,
                                                src_words, tgt_words, inserted_words,
                                                deleted_words, collections)
        ops[ES_TRACE] = trace_summary

    if config.get('logger.enable_bleu', True):
        avg_bleu = add_extra_summary_avg_bleu(vocab_i2s, decoder_output, tgt_words, collections)
        ops[ES_BLEU] = avg_bleu

    return ops


def add_extra_summary_trace(vocab_i2s, decoder_output, src_words, tgt_words, inserted_words, deleted_words,
                            collections=None):
    pred_tokens = decoder.str_tokens(decoder_output, vocab_i2s)
    pred_len = decoder.seq_length(decoder_output)

    tgt_tokens = vocab_i2s.lookup(tgt_words)
    tgt_len = length_pre_embedding(tgt_words)

    trace_summary = get_trace_summary(vocab_i2s, pred_tokens, tgt_tokens, src_words, inserted_words, deleted_words,
                                      pred_len, tgt_len)
    tf.summary.text('trace', trace_summary, collections)

    return trace_summary


def add_extra_summary_avg_bleu(vocab_i2s, decoder_output, tgt_words, collections=None):
    pred_tokens = decoder.str_tokens(decoder_output, vocab_i2s)
    pred_len = decoder.seq_length(decoder_output)

    tgt_tokens = vocab_i2s.lookup(tgt_words)
    tgt_len = length_pre_embedding(tgt_words)

    avg_bleu = get_avg_bleu_smmary(tgt_tokens, pred_tokens, tgt_len, pred_len)
    tf.summary.scalar('bleu', avg_bleu, collections)

    return avg_bleu


def model_fn(features, mode, config, embedding_matrix, vocab_tables):
    if mode == tf.estimator.ModeKeys.PREDICT:
        base_words, src_words, tgt_words, inserted_words, deleted_words = features
    else:
        src_words, tgt_words, inserted_words, deleted_words = features
        base_words = src_words

    vocab_s2i = vocab_tables[vocab.STR_TO_INT]
    vocab_i2s = vocab_tables[vocab.INT_TO_STR]

    vocab.init_embeddings(embedding_matrix)

    train_decoder_output, infer_decoder_output, \
    gold_dec_out, gold_dec_out_len = editor.editor_train(
        base_words, src_words, tgt_words, inserted_words, deleted_words, embedding_matrix, vocab_s2i,
        config.editor.hidden_dim, config.editor.agenda_dim, config.editor.edit_dim,
        config.editor.encoder_layers, config.editor.decoder_layers, config.editor.attention_dim,
        config.editor.beam_width,
        config.editor.max_sent_length, config.editor.dropout_keep, config.editor.lamb_reg,
        config.editor.norm_eps, config.editor.norm_max, config.editor.kill_edit,
        config.editor.draw_edit, config.editor.use_swap_memory, config.get('editor.use_beam_decoder', False)
    )

    loss = optimizer.loss(train_decoder_output, gold_dec_out, gold_dec_out_len)
    train_op, gradients_norm = optimizer.train(loss, config.optim.learning_rate, config.optim.max_norm_observe_steps)

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('grad_norm', gradients_norm)
        ops = add_extra_summary(config, vocab_i2s, train_decoder_output,
                                src_words, tgt_words, inserted_words, deleted_words,
                                ['extra'])

        return tf.estimator.EstimatorSpec(
            mode,
            train_op=train_op,
            loss=loss,
            training_hooks=[
                get_train_extra_summary_writer(config),
                get_extra_summary_logger(ops, config),
                get_profiler_hook(config)
            ]
        )

    elif mode == tf.estimator.ModeKeys.EVAL:
        ops = add_extra_summary(config, vocab_i2s, train_decoder_output,
                                src_words, tgt_words, inserted_words, deleted_words,
                                ['extra'])

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            evaluation_hooks=[get_extra_summary_logger(ops, config)],
            eval_metric_ops={'bleu': tf_metrics.streaming_mean(ops[ES_BLEU])}
        )

    elif mode == tf.estimator.ModeKeys.PREDICT:
        lengths = decoder.seq_length(infer_decoder_output)
        tokens = decoder.str_tokens(infer_decoder_output, vocab_i2s)
        preds = {
            'str_tokens': tokens,
            'sample_id': decoder.sample_id(infer_decoder_output),
            'lengths': lengths,
            'joined': metrics.join_tokens(tokens, lengths),
        }

        return tf.estimator.EstimatorSpec(
            mode,
            predictions=preds
        )
