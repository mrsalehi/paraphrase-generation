import tensorflow as tf
import tensorflow.contrib.metrics as tf_metrics

tf.logging.set_verbosity(tf.logging.INFO)

from models.common import vocab, metrics
from models.common.sequence import length_pre_embedding
from models.neural_editor import editor, optimizer, decoder


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
            output_str += 'SOURCE: %s\n' % str(src, encoding='utf8')
            output_str += 'INSERT: [ %s ]\n' % str(iw, encoding='utf8')
            output_str += 'DELETE: [ %s ]\n' % str(dw, encoding='utf8')
            output_str += 'TARGET: %s\n' % str(tgt, encoding='utf8')
            output_str += '---\n%s\n---\n\n' % str(pred, encoding='utf8')
            output_str += '=========================\n\n'

        return output_str

    def print_bleu_score(bleu_result):
        return 'BLEU: %s' % bleu_result

    return tf.train.LoggingTensorHook(
        tensors=tensors,
        every_n_iter=every_n_step,
        formatter=lambda t: '\n'.join([print_pred_summary(t['pred']), print_bleu_score(t['bleu'])])
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


def add_extra_summary(vocab_i2s, decoder_output, src_words, tgt_words, inserted_words, deleted_words, collections=None):
    pred_tokens = decoder.str_tokens(decoder_output, vocab_i2s)
    pred_len = decoder.seq_length(decoder_output)

    tgt_tokens = vocab_i2s.lookup(tgt_words)
    tgt_len = length_pre_embedding(tgt_words)

    avg_bleu = get_avg_bleu_smmary(tgt_tokens, pred_tokens, tgt_len, pred_len)
    tf.summary.scalar('bleu', avg_bleu, collections)

    trace_summary = get_trace_summary(vocab_i2s, pred_tokens, tgt_tokens, src_words, inserted_words, deleted_words,
                                      pred_len, tgt_len)
    tf.summary.text('trace', trace_summary, collections)

    return avg_bleu, trace_summary


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
        config.editor.draw_edit, config.editor.use_swap_memory
    )

    loss = optimizer.loss(train_decoder_output, gold_dec_out, gold_dec_out_len)
    train_op, gradients_norm = optimizer.train(loss, config.optim.learning_rate, config.optim.max_norm_observe_steps)

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('grad_norm', gradients_norm)
        avg_bleu, pred_summary = add_extra_summary(vocab_i2s, infer_decoder_output,
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
        avg_bleu, pred_summary = add_extra_summary(vocab_i2s, infer_decoder_output,
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
