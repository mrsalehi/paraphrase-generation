import numpy as np
import tensorflow as tf
import tensorflow.contrib.metrics as tf_metrics

from models.common.sequence import length_pre_embedding
from models.im_all_transformer.editor import Editor
from models.im_all_transformer.transformer.embedding_layer import EmbeddingSharedWeights
from models.neural_editor.model import get_profiler_hook, ES_BLEU, get_train_extra_summary_writer, ES_TRACE

tf.logging.set_verbosity(tf.logging.INFO)

from models.common import vocab, metrics, sequence
from models.im_all_transformer import decoder, optimizer


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


def get_avg_bleu(tgt_tokens, pred_tokens, tgt_len, pred_len):
    if pred_tokens.shape.ndims > 2:
        best_tokens = pred_tokens[:, :, 0]
        best_len = pred_len[:, 0]
    else:
        best_tokens = pred_tokens
        best_len = pred_len

    bleu_score = metrics.create_bleu_metric_ops(tgt_tokens, best_tokens, tgt_len, best_len)
    avg_bleu = tf.reduce_mean(bleu_score)

    return avg_bleu


def get_trace(pred_tokens, tgt_tokens,
              src_words, inserted_words, deleted_words,
              pred_len, tgt_len):
    vocab_i2s = vocab.get_vocab_lookup_tables()[vocab.INT_TO_STR]

    if pred_tokens.shape.ndims > 2:
        pred_joined = metrics.join_beams(pred_tokens, pred_len)
    else:
        pred_joined = metrics.join_tokens(pred_tokens, pred_len)

    tgt_joined = metrics.join_tokens(tgt_tokens, tgt_len)
    src_joined = metrics.join_tokens(vocab_i2s.lookup(src_words), length_pre_embedding(src_words))
    iw_joined = metrics.join_tokens(vocab_i2s.lookup(inserted_words), length_pre_embedding(inserted_words), ', ')
    dw_joined = metrics.join_tokens(vocab_i2s.lookup(deleted_words), length_pre_embedding(deleted_words), ', ')

    return tf.concat([src_joined, iw_joined, dw_joined, tgt_joined, pred_joined], axis=1)


def add_extra_summary_trace(pred_tokens, pred_len,
                            base_words, output_words,
                            src_words, tgt_words, inserted_words, deleted_words,
                            collections=None):
    vocab_i2s = vocab.get_vocab_lookup_tables()[vocab.INT_TO_STR]

    tgt_tokens = vocab_i2s.lookup(tgt_words)
    tgt_len = length_pre_embedding(tgt_words)

    trace_summary = get_trace(pred_tokens, tgt_tokens, src_words, inserted_words, deleted_words,
                              pred_len, tgt_len)
    tf.summary.text('trace', trace_summary, collections)

    return trace_summary


def add_extra_summary_avg_bleu(hypo_tokens, hypo_len, ref_words, collections=None):
    vocab_i2s = vocab.get_vocab_lookup_tables()[vocab.INT_TO_STR]

    ref_tokens = vocab_i2s.lookup(ref_words)
    ref_len = length_pre_embedding(ref_words)

    avg_bleu = get_avg_bleu(ref_tokens, hypo_tokens, ref_len, hypo_len)
    tf.summary.scalar('bleu', avg_bleu, collections)

    return avg_bleu


def add_extra_summary(config,
                      decoded_ids, decoded_length,
                      base_word, output_words, src_words, tgt_words, inserted_words, deleted_words,
                      collections=None):
    pred_tokens = decoder.str_tokens(decoded_ids)
    pred_len = decoded_length

    ops = {}

    if config.get('logger.enable_trace', False):
        trace_summary = add_extra_summary_trace(pred_tokens, pred_len,
                                                base_word, output_words, src_words, tgt_words, inserted_words,
                                                deleted_words, collections)
        ops[ES_TRACE] = trace_summary

    if config.get('logger.enable_bleu', True):
        avg_bleu = add_extra_summary_avg_bleu(pred_tokens, pred_len, output_words, collections)
        ops[ES_BLEU] = avg_bleu

    return ops


def model_fn(features, mode, config, embedding_matrix, vocab_tables):
    if mode == tf.estimator.ModeKeys.PREDICT:
        base_words, _, src_words, tgt_words, inserted_words, commong_words = features
        output_words = tgt_words
    else:
        base_words, output_words, src_words, tgt_words, inserted_words, commong_words = features

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    tf.add_to_collection('is_training', is_training)

    if mode != tf.estimator.ModeKeys.TRAIN:
        config.put('editor.enable_dropout', False)
        config.put('editor.dropout_keep', 1.0)
        config.put('editor.dropout', 0.0)

        config.put('editor.transformer.enable_dropout', False)
        config.put('editor.transformer.layer_postprocess_dropout', 0.0)
        config.put('editor.transformer.attention_dropout', 0.0)
        config.put('editor.transformer.relu_dropout', 0.0)

    vocab.init_embeddings(embedding_matrix)
    EmbeddingSharedWeights.init_from_embedding_matrix()

    editor_model = Editor(config)
    logits, beam_prediction = editor_model(
        base_words, src_words, tgt_words, inserted_words, commong_words,
        output_words
    )

    targets = decoder.prepare_decoder_output(
        output_words,
        sequence.length_pre_embedding(output_words)
    )
    target_lengths = sequence.length_pre_embedding(targets)

    vocab_size = embedding_matrix.shape[0]
    loss, weights = optimizer.padded_cross_entropy_loss(
        logits, targets, target_lengths,
        config.optim.label_smoothing,
        vocab_size
    )

    train_op = optimizer.get_train_op(loss, config)

    tf.logging.info("Trainable variable")
    for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        tf.logging.info(str(i))

    tf.logging.info("Num of Trainable parameters")
    tf.logging.info(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    if mode == tf.estimator.ModeKeys.TRAIN:
        decoded_ids = decoder.logits_to_decoded_ids(logits)
        ops = add_extra_summary(config, decoded_ids, target_lengths,
                                base_words, output_words,
                                src_words, tgt_words,
                                inserted_words, commong_words,
                                collections=['extra'])

        hooks = [
            get_train_extra_summary_writer(config),
            get_extra_summary_logger(ops, config),
        ]

        if config.get('logger.enable_profiler', False):
            hooks.append(get_profiler_hook(config))

        return tf.estimator.EstimatorSpec(
            mode,
            train_op=train_op,
            loss=loss,
            training_hooks=hooks
        )

    elif mode == tf.estimator.ModeKeys.EVAL:
        decoded_ids = decoder.logits_to_decoded_ids(logits)
        ops = add_extra_summary(config, decoded_ids, target_lengths,
                                base_words, output_words,
                                src_words, tgt_words,
                                inserted_words, commong_words,
                                collections=['extra'])

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            evaluation_hooks=[get_extra_summary_logger(ops, config)],
            eval_metric_ops={'bleu': tf_metrics.streaming_mean(ops[ES_BLEU])}
        )

    elif mode == tf.estimator.ModeKeys.PREDICT:
        decoded_ids, decoded_lengths, scores = beam_prediction
        tokens = decoder.str_tokens(decoded_ids)

        preds = {
            'str_tokens': tf.transpose(tokens, [0, 2, 1]),
            'decoded_ids': tf.transpose(decoded_ids, [0, 2, 1]),
            'lengths': decoded_lengths,
            'joined': metrics.join_tokens(tokens, decoded_lengths)
        }

        tmee_attentions = tf.get_collection('TransformerMicroEditExtractor_Attentions')
        if len(tmee_attentions) > 0:
            preds.update({
                'tmee_attentions_st_enc_self': tmee_attentions[0][0],
                'tmee_attentions_st_dec_self': tmee_attentions[0][1],
                'tmee_attentions_st_dec_enc': tmee_attentions[0][2],
                'tmee_attentions_ts_enc_self': tf.zeros_like(tmee_attentions[0][0]),
                'tmee_attentions_ts_dec_self': tf.zeros_like(tmee_attentions[0][1]),
                'tmee_attentions_ts_dec_enc': tf.zeros_like(tmee_attentions[0][2]),
                'src_words': src_words,
                'tgt_words': tgt_words,
                'base_words': base_words,
                'output_words': output_words
            })

        return tf.estimator.EstimatorSpec(
            mode,
            predictions=preds
        )
