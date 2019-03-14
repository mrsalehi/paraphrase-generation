import numpy as np
import tensorflow as tf

from models import neural_editor
from models.common import vocab, sequence
from models.neural_editor import agenda
from models.neural_editor import decoder
from models.neural_editor import edit_encoder
from models.neural_editor import encoder
from test.test_input_fn import EMBED_DIM, NUM_EPOCH, BATCH_SIZE

EDIT_DIM = 3


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

            while True:
                try:
                    dec_inputs, dec_outputs, tgt_len, dil, dol, start_token_id, stop_token_id, dec_outputs_last, tgt = sess.run(
                        [dec_inputs, dec_outputs, tgt_len, dec_inputs_len, dec_outputs_len, start_token_id,
                         stop_token_id,
                         dec_outputs_last, tgt])

                    assert list(dil) == list(dol) == list(tgt_len + 1)
                    assert list(dec_inputs[:, 0]) == list(np.ones_like(dec_inputs[:, 0]) * start_token_id)
                    assert list(dec_outputs_last) == list(np.ones_like(dec_outputs_last) * stop_token_id)
                except:
                    break


def test_decoder_train(dataset_file, embedding_file):
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
        (src, tgt, inw, dlw), _ = iter.get_next()

        src_len = sequence.length_pre_embedding(src)

        tgt_len = sequence.length_pre_embedding(tgt)

        dec_inputs = decoder.prepare_decoder_inputs(tgt, start_token_id)
        dec_outputs = decoder.prepare_decoder_output(tgt, tgt_len, stop_token_id, pad_token_id)

        dec_inputs_len = sequence.length_pre_embedding(dec_inputs)
        dec_outputs_len = sequence.length_pre_embedding(dec_outputs)

        batch_size = tf.shape(src)[0]
        edit_vector = edit_encoder.random_noise_encoder(batch_size, EDIT_DIM, 14.0)

        embedding = tf.get_variable('embeddings', shape=embed_matrix.shape,
                                    initializer=tf.constant_initializer(embed_matrix))

        src_embd = tf.nn.embedding_lookup(embedding, src)
        src_sent_embeds, final_states = encoder.source_sent_encoder(src_embd, src_len, 20, 3, 0.8)

        agn = agenda.linear(final_states, edit_vector, 4)

        dec_out = decoder.train_decoder(
            agn, embedding,
            dec_inputs,
            src_sent_embeds,
            tf.nn.embedding_lookup(embedding, inw),
            tf.nn.embedding_lookup(embedding, dlw),
            dec_inputs_len, src_len, sequence.length_pre_embedding(inw), sequence.length_pre_embedding(dlw),
            5, 20, 3, False
        )

        # eval_dec_out = decoder.greedy_eval_decoder(
        #     agn, embedding,
        #     start_token_id, stop_token_id,
        #     src_sent_embeds,
        #     tf.nn.embedding_lookup(embedding, inw),
        #     tf.nn.embedding_lookup(embedding, dlw),
        #     src_len, sequence.length_pre_embedding(inw), sequence.length_pre_embedding(dlw),
        #     5, 20, 3, 40
        # )

        eval_dec_out = decoder.beam_eval_decoder(
            agn, embedding,
            start_token_id, stop_token_id,
            src_sent_embeds,
            tf.nn.embedding_lookup(embedding, inw),
            tf.nn.embedding_lookup(embedding, dlw),
            src_len, sequence.length_pre_embedding(inw), sequence.length_pre_embedding(dlw),
            5, 20, 3, 40
        )

        # saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
        # s = tf.summary.FileWriter('data/an')
        # s.add_graph(g)
        #
        # all_print = tf.get_collection('print')

        an, final_states, len = dec_out
        stacked = decoder.attention_score(dec_out)

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
            sess.run(iter.initializer)

            print(sess.run([eval_dec_out]))

            # (src_embd, src_len, tgt, inw, dlw) = sess.run((src_sent_embeds, src_len,tgt, inw, dlw))

            # o = sess.run([an, stacked, len])
            # t = sess.run(tgt)
            # tl = sess.run(tgt_len)
            # do = sess.run(dec_outputs)
            # di = sess.run(dec_inputs)
            # dol = sess.run(dec_outputs_len)
            # dil = sess.run(dec_inputs_len)
            # tgt_len = sess.run(tgt_len)

            # assert list(dil) == list(dol) == list(tgt_len + 1)
            # assert list(di[:, 0]) == list(np.ones_like(di[:, 0]) * start_token_id)
            # print(o[1][0].shape)

            # save_path = saver.save(sess, "data/an/model")
            # print(save_path)
            # fucker = tf.get_default_graph().get_tensor_by_name(
            #     "decoder_1/AttentionAugmentRNNCellZeroState/AttentionWrapperZeroState/zeros_4/Identity:0")
            # print(sess.run(tf.shape(fucker)))
            # # print(tf.get_default_graph().get_operation_by_name())
            # print(fucker)
            # for i in all_print:
            #     try:
            #         print(sess.run(i))
            #     except Exception as e:
            #         print(e)
            #         pass
            # o2 = sess.run(eval_dec_out)
