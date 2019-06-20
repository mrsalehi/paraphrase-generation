import tensorflow as tf
import tensorflow.contrib.metrics as tf_metrics

from models.im_all_transformer.transformer.embedding_layer import EmbeddingSharedWeights
from models.im_attn_ee_rnn_attn_dec_pg.model import add_extra_summary, get_extra_summary_logger, add_decoder_attention
from models.neural_editor.model import get_profiler_hook, ES_BLEU, get_train_extra_summary_writer

tf.logging.set_verbosity(tf.logging.INFO)

from models.common import vocab, metrics, sequence
from models.im_all_transformer import editor, decoder, optimizer


def model_fn(features, mode, config, embedding_matrix, vocab_tables):
    if mode == tf.estimator.ModeKeys.PREDICT:
        base_words, _, src_words, tgt_words, inserted_words, deleted_words = features
        output_words = tgt_words
    else:
        base_words, output_words, src_words, tgt_words, inserted_words, deleted_words = features

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    tf.add_to_collection('is_training', is_training)

    if mode != tf.estimator.ModeKeys.TRAIN:
        config.put('editor.enable_dropout', False)
        config.put('editor.dropout_keep', 1.0)

    vocab.init_embeddings(embedding_matrix)
    EmbeddingSharedWeights.init_from_embedding_matrix()

    logits, beam_decoded_ids, beam_decoded_lengths = editor.editor_train(
        base_words, output_words,
        src_words, tgt_words, inserted_words, deleted_words,
        config
    )

    decoder_targets = decoder.prepare_decoder_output(
        output_words,
        sequence.length_pre_embedding(output_words)
    )
    decoder_target_len = sequence.length_pre_embedding(decoder_targets)

    vocab_size = embedding_matrix.shape[0]
    loss, weights = optimizer.padded_cross_entropy_loss(
        logits, decoder_targets, decoder_target_len,
        config.optim.label_smoothing,
        vocab_size
    )

    train_op = optimizer.get_train_op(loss, config)

    tf.logging.info("Trainable variable")
    for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        tf.logging.info(str(i))

    if mode == tf.estimator.ModeKeys.TRAIN:
        # tf.summary.scalar('grad_norm', gradients_norm)
        # ops = add_extra_summary(config, vocab_i2s, train_decoder_output,
        #                         base_words, output_words,
        #                         src_words, tgt_words,
        #                         inserted_words, deleted_words, oov,
        #                         vocab_size, collections=['extra'])

        # hooks = [
        #     get_train_extra_summary_writer(config),
        #     get_extra_summary_logger(ops, config),
        # ]

        # if config.get('logger.enable_profiler', False):
        #     hooks.append(get_profiler_hook(config))

        return tf.estimator.EstimatorSpec(
            mode,
            train_op=train_op,
            loss=loss,
            # training_hooks=hooks
        )

    elif mode == tf.estimator.ModeKeys.EVAL:
        # ops = add_extra_summary(config, vocab_i2s, train_decoder_output,
        #                         base_words, output_words,
        #                         src_words, tgt_words,
        #                         inserted_words, deleted_words, oov,
        #                         vocab_size, collections=['extra'])

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            # evaluation_hooks=[get_extra_summary_logger(ops, config)],
            # eval_metric_ops={'bleu': tf_metrics.streaming_mean(ops[ES_BLEU])}
        )
    #
    elif mode == tf.estimator.ModeKeys.PREDICT:
        lengths = beam_decoded_lengths
        tokens = decoder.str_tokens(beam_decoded_ids)
        # attns_weight = tf.get_collection('attns_weight')

        preds = {
            # 'str_tokens': tokens,
            'decoded_ids': beam_decoded_ids,
            'lengths': lengths,
            'joined': metrics.join_tokens(tokens, lengths),
            # 'attns_weight_0': attns_weight[0],
            # 'attns_weight_1': attns_weight[1]
        }

        # add_decoder_attention(config, preds)

        return tf.estimator.EstimatorSpec(
            mode,
            predictions=preds
        )
