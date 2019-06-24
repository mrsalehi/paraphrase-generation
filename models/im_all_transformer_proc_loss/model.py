import tensorflow as tf
import tensorflow.contrib.metrics as tf_metrics

from models.common.sequence import length_pre_embedding
from models.im_all_transformer.editor import Editor
from models.im_all_transformer.model import add_extra_summary, get_extra_summary_logger
from models.im_all_transformer.transformer.embedding_layer import EmbeddingSharedWeights
from models.neural_editor.model import get_profiler_hook, ES_BLEU, get_train_extra_summary_writer, ES_TRACE

tf.logging.set_verbosity(tf.logging.INFO)

from models.common import vocab, metrics, sequence
from models.im_all_transformer import decoder, optimizer


def calculate_loss(logits, output_words, input_words, tgt_words, label_smoothing,
                   vocab_size):
    gold = decoder.prepare_decoder_output(
        output_words,
        sequence.length_pre_embedding(output_words)
    )
    gold_len = sequence.length_pre_embedding(gold)

    gold_input = decoder.prepare_decoder_output(
        input_words,
        sequence.length_pre_embedding(input_words)
    )
    gold_input_len = sequence.length_pre_embedding(gold_input)

    gold_tgt = decoder.prepare_decoder_output(
        tgt_words,
        sequence.length_pre_embedding(tgt_words)
    )
    gold_tgt_len = sequence.length_pre_embedding(gold_tgt)

    main_loss, _ = optimizer.padded_cross_entropy_loss(logits, gold, gold_len, label_smoothing, vocab_size)

    input_loss, _ = optimizer.padded_cross_entropy_loss(logits, gold_input, gold_input_len, label_smoothing, vocab_size)
    tgt_loss, _ = optimizer.padded_cross_entropy_loss(logits, gold_tgt, gold_tgt_len, label_smoothing, vocab_size)

    total_loss = main_loss - 1./50 * input_loss - 1./30 * tgt_loss

    return total_loss


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
    loss = calculate_loss(logits, output_words, base_words, tgt_words,
                          config.optim.label_smoothing, vocab_size)

    train_op = optimizer.get_train_op(loss, config)

    tf.logging.info("Trainable variable")
    for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        tf.logging.info(str(i))

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
            'joined': metrics.join_tokens(tokens, decoded_lengths),
        }

        return tf.estimator.EstimatorSpec(
            mode,
            predictions=preds
        )
