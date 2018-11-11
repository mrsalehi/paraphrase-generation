import numpy as np

import tensorflow as tf

from models.common import vocab, sequence
from models import neural_editor
from models.neural_editor import decoder

from test.test_input_fn import embedding_file, dataset_file, dataset, EMBED_DIM, NUM_EPOCH, BATCH_SIZE


def test_decoder_prepares(dataset_file, embedding_file):
    with tf.Graph().as_default():
        d_fn, gold_dataset = dataset_file
        e_fn, gold_embeds = embedding_file

        v, embed_matrix = vocab.read_word_embeddings(e_fn, EMBED_DIM)
        vocab_lookup = vocab.get_vocab_lookup(v)

        stop_token = tf.constant(bytes(vocab.STOP_TOKEN, encoding='utf8'), dtype=tf.string)
        stop_token_id = vocab_lookup.lookup(stop_token)

        start_token = tf.constant(bytes(vocab.START_TOKEN, encoding='utf8'), dtype=tf.string)
        start_token_id = vocab_lookup.lookup(start_token)

        pad_token = tf.constant(bytes(vocab.PAD_TOKEN, encoding='utf8'), dtype=tf.string)
        pad_token_id = vocab_lookup.lookup(pad_token)

        dataset = neural_editor.input_fn(d_fn, vocab_lookup, BATCH_SIZE, NUM_EPOCH)
        iter = dataset.make_initializable_iterator()
        (_, tgt, _, _), _ = iter.get_next()

        tgt_len = sequence.length_pre_embedding(tgt)

        dec_inputs = decoder.prepare_decoder_inputs(tgt, start_token_id)
        dec_outputs = decoder.prepare_decoder_output(tgt, tgt_len, stop_token_id, pad_token_id)

        dec_inputs_len = sequence.length_pre_embedding(dec_inputs)
        dec_outputs_len = sequence.length_pre_embedding(dec_outputs)

        dec_outputs_last = sequence.last_relevant(tf.expand_dims(dec_outputs, 2), dec_outputs_len)
        dec_outputs_last = tf.squeeze(dec_outputs_last)

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
            sess.run(iter.initializer)

            dec_inputs, dec_outputs, tgt_len, dil, dol, start_token_id, stop_token_id, dec_outputs_last, tgt = sess.run(
                [dec_inputs, dec_outputs, tgt_len, dec_inputs_len, dec_outputs_len, start_token_id, stop_token_id,
                 dec_outputs_last, tgt])

            assert list(dil) == list(dol) == list(tgt_len + 1)
            assert list(dec_inputs[:, 0]) == list(np.ones_like(dec_inputs[:, 0]) * start_token_id)
            assert list(dec_outputs_last) == list(np.ones_like(dec_outputs_last) * stop_token_id)
