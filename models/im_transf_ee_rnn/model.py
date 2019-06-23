import tensorflow as tf
import tensorflow.contrib.metrics as tf_metrics

from models.im_attn_ee_rnn_attn_dec_pg.model import add_extra_summary, get_extra_summary_logger, add_decoder_attention
from models.neural_editor.model import get_profiler_hook, ES_BLEU, get_train_extra_summary_writer

tf.logging.set_verbosity(tf.logging.INFO)

from models.common import vocab, metrics
from models.neural_editor import optimizer
from models.im_transf_ee_rnn import editor
from models.im_attn_ee_rnn_attn_dec_pg import decoder


def model_fn(features, mode, config, embedding_matrix, vocab_tables):
    if mode == tf.estimator.ModeKeys.PREDICT:
        base_words, extended_base_words, \
        _, _, \
        src_words, tgt_words, \
        inserted_words, deleted_words, \
        oov = features
        output_words = extended_output_words = tgt_words
    else:
        base_words, extended_base_words, \
        output_words, extended_output_words, \
        src_words, tgt_words, \
        inserted_words, deleted_words, \
        oov = features

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    tf.add_to_collection('is_training', is_training)

    if mode != tf.estimator.ModeKeys.TRAIN:
        config.put('editor.enable_dropout', False)
        config.put('editor.dropout_keep', 1.0)

    vocab_i2s = vocab_tables[vocab.INT_TO_STR]
    vocab.init_embeddings(embedding_matrix)

    vocab_size = len(vocab_tables[vocab.RAW_WORD2ID])

    train_decoder_output, infer_decoder_output, \
    gold_dec_out, gold_dec_out_len = editor.editor_train(
        base_words, extended_base_words, output_words, extended_output_words,
        src_words, tgt_words, inserted_words, deleted_words, oov,
        vocab_size,
        config.editor.hidden_dim, config.editor.agenda_dim, config.editor.edit_dim,
        config.editor.edit_enc.micro_ev_dim, config.editor.edit_enc.num_heads,
        config.editor.encoder_layers, config.editor.decoder_layers, config.editor.attention_dim,
        config.editor.beam_width,
        config.editor.edit_enc.transformer,
        config.editor.edit_enc.wa_hidden_dim, config.editor.edit_enc.wa_hidden_layer,
        config.editor.edit_enc.meve_hidden_dim, config.editor.edit_enc.meve_hidden_layer,
        config.editor.max_sent_length, config.editor.dropout_keep, config.editor.lamb_reg,
        config.editor.norm_eps, config.editor.norm_max, config.editor.kill_edit,
        config.editor.draw_edit, config.editor.use_swap_memory,
        config.get('editor.use_beam_decoder', False), config.get('editor.enable_dropout', False),
        config.get('editor.no_insert_delete_attn', False), config.get('editor.enable_vae', True)
    )

    loss = optimizer.loss(train_decoder_output, gold_dec_out, gold_dec_out_len)
    train_op, gradients_norm = optimizer.train(loss, config.optim.learning_rate, config.optim.max_norm_observe_steps)

    tf.logging.info("Trainable variable")
    for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        tf.logging.info(str(i))

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('grad_norm', gradients_norm)
        ops = add_extra_summary(config, vocab_i2s, train_decoder_output,
                                base_words, output_words,
                                src_words, tgt_words,
                                inserted_words, deleted_words, oov,
                                vocab_size, collections=['extra'])

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
        ops = add_extra_summary(config, vocab_i2s, train_decoder_output,
                                base_words, output_words,
                                src_words, tgt_words,
                                inserted_words, deleted_words, oov,
                                vocab_size, collections=['extra'])

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            evaluation_hooks=[get_extra_summary_logger(ops, config)],
            eval_metric_ops={'bleu': tf_metrics.streaming_mean(ops[ES_BLEU])}
        )

    elif mode == tf.estimator.ModeKeys.PREDICT:
        lengths = decoder.seq_length(infer_decoder_output)
        tokens = decoder.str_tokens(infer_decoder_output, vocab_i2s, vocab_size, oov)
        # attns_weight = tf.get_collection('attns_weight')

        preds = {
            'str_tokens': tokens,
            'sample_id': decoder.sample_id(infer_decoder_output),
            'lengths': lengths,
            'joined': metrics.join_tokens(tokens, lengths),
        }

        add_decoder_attention(config, preds)

        return tf.estimator.EstimatorSpec(
            mode,
            predictions=preds
        )
