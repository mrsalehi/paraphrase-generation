import numpy as np
import tensorflow as tf
import tensorflow.contrib.metrics as tf_metrics

from models.im_all_transformer.editor import Editor
from models.im_all_transformer.model import add_extra_summary, get_extra_summary_logger
from models.im_all_transformer.transformer.embedding_layer import EmbeddingSharedWeights
from models.neural_editor.model import get_profiler_hook, ES_BLEU, get_train_extra_summary_writer

tf.logging.set_verbosity(tf.logging.INFO)

from models.common import vocab, metrics, sequence
from models.im_all_transformer import decoder
from models.im_all_transformer_straight_attn_pq_sampling import optimizer


def model_fn(features, mode, config, embedding_matrix, vocab_tables):
    if mode == tf.estimator.ModeKeys.PREDICT:
        base_words, _, src_words, tgt_words, inserted_words, common_words = features
        output_words = tgt_words
        batch_size = tf.shape(base_words)[0]
        loss_weighs = tf.ones(shape=[batch_size], dtype=tf.float32)
    else:
        inputs_, loss_weighs = features
        base_words, output_words, src_words, tgt_words, inserted_words, common_words = inputs_

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
        base_words, src_words, tgt_words, inserted_words, common_words,
        output_words
    )

    targets = decoder.prepare_decoder_output(
        output_words,
        sequence.length_pre_embedding(output_words)
    )
    target_lengths = sequence.length_pre_embedding(targets)

    vocab_size = embedding_matrix.shape[0]
    loss, weights = optimizer.padded_cross_entropy_loss(
        logits, targets, target_lengths, loss_weighs,
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
                                inserted_words, common_words,
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
                                inserted_words, common_words,
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
                'tmee_attentions_ts_enc_self': tmee_attentions[1][0],
                'tmee_attentions_ts_dec_self': tmee_attentions[1][1],
                'tmee_attentions_ts_dec_enc': tmee_attentions[1][2],
                'src_words': src_words,
                'tgt_words': tgt_words,
                'base_words': base_words,
                'output_words': output_words
            })

        return tf.estimator.EstimatorSpec(
            mode,
            predictions=preds
        )
