import os
import tensorflow as tf
import numpy as np

import models.common.vocab as vocab
import models.common.sequence as sequence
import models.neural_editor as neural_editor
import models.neural_editor.edit_encoder as ev

from test.test_input_fn import embedding_file, dataset_file, dataset, EMBED_DIM, NUM_EPOCH, BATCH_SIZE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

HIDDEN_DIM = 20
NUM_LAYER = 1


# def test_word_aggregator(dataset_file, embedding_file):
#     with tf.Graph().as_default():
#         d_fn, gold_dataset = dataset_file
#         e_fn, gold_embeds = embedding_file
#
#         v, embed_matrix = vocab.read_word_embeddings(e_fn, EMBED_DIM)
#         vocab_lookup = vocab.get_vocab_lookup(v)
#
#         dataset = neural_editor.input_fn(d_fn, vocab_lookup, BATCH_SIZE, NUM_EPOCH)
#
#         embedding = tf.get_variable('embeddings', shape=embed_matrix.shape,
#                                     initializer=tf.constant_initializer(embed_matrix))
#
#         iter = dataset.make_initializable_iterator()
#         (_, _, src, _), _ = iter.get_next()
#
#         src_len = sequence.length_pre_embedding(src)
#         src_embd = tf.nn.embedding_lookup(embedding, src)
#
#         output = ev.word_aggregator(src_embd, src_len, HIDDEN_DIM, NUM_LAYER)
#
#         with tf.Session() as sess:
#             sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
#             sess.run(iter.initializer)
#
#             while True:
#                 try:
#                     oeo, o_src, o_src_len, o_src_embd = sess.run([output, src, src_len, src_embd])
#                     assert oeo.shape == (BATCH_SIZE, o_src_len.max(), HIDDEN_DIM)
#                 except:
#                     break
#
#
# def test_context_encoder(dataset_file, embedding_file):
#     with tf.Graph().as_default():
#         d_fn, gold_dataset = dataset_file
#         e_fn, gold_embeds = embedding_file
#
#         v, embed_matrix = vocab.read_word_embeddings(e_fn, EMBED_DIM)
#         vocab_lookup = vocab.get_vocab_lookup(v)
#
#         dataset = neural_editor.input_fn(d_fn, vocab_lookup, BATCH_SIZE, NUM_EPOCH)
#
#         embedding = tf.get_variable('embeddings', shape=embed_matrix.shape,
#                                     initializer=tf.constant_initializer(embed_matrix))
#
#         iter = dataset.make_initializable_iterator()
#         (_, _, src, _), _ = iter.get_next()
#
#         src_len = sequence.length_pre_embedding(src)
#         src_embd = tf.nn.embedding_lookup(embedding, src)
#
#         output = ev.context_encoder(src_embd, src_len, HIDDEN_DIM, NUM_LAYER)
#
#         with tf.Session() as sess:
#             sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
#             sess.run(iter.initializer)
#
#             while True:
#                 try:
#                     oeo, o_src, o_src_len, o_src_embd = sess.run([output, src, src_len, src_embd])
#                     assert oeo.shape == (BATCH_SIZE, o_src_len.max(), HIDDEN_DIM)
#                 except:
#                     break


def test_sample_weight():
    tf.enable_eager_execution()
    with tf.Graph().as_default():
        batch_size = tf.placeholder(tf.int32, shape=())
        w = ev.sample_weight_tf(100.0, 2, batch_size)

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
            o = sess.run(w, feed_dict={batch_size: 4})

            # print(o)


def test_sample_orthonormal_to():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            with sess.as_default():
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
                # sess.as_default()
                # tf.enable_eager_execution()
                dim = 2
                # batch_size = tf.placeholder(tf.int32, shape=())
                batch_size = 4

                m = tf.ones((batch_size, 2))
                munorm = tf.norm(m, axis=1, keepdims=True)
                munorm = tf.tile(munorm, [1, dim])
                res = ev.sample_orthonormal_to(m / munorm, dim, batch_size)
                # print(sess.run(res))

        # with tf.Session() as sess:
        #     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
        #     o = sess.run(res, feed_dict={batch_size: 4})
        #
        #     print(o)


def test_add_norm_noise():
    with tf.Graph().as_default():
        dim = 2
        batch_size = tf.placeholder(tf.int32, shape=())
        m = tf.ones((batch_size, 2))
        munorm = tf.norm(m, axis=1, keepdims=True)
        munorm = tf.tile(munorm, [1, dim])
        munoise = ev.add_norm_noise(munorm, 0.1, 14, batch_size)

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
            o, norm = sess.run([munoise, munorm], feed_dict={batch_size: 4})

            assert np.abs(o - norm).mean() < 0.1

def test_sample_vMF():
    with tf.Graph().as_default():
        dim = 2
        kappa = 100.
        norm_eps = 0.1
        norm_max = 14

        batch_size = tf.placeholder(tf.int32, shape=())
        m = tf.ones((batch_size, dim))*2
        noisy = ev.sample_vMF(m, kappa, norm_eps, norm_max)

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
            o = sess.run(noisy, feed_dict={batch_size: 4})
            print()
            print(o)
