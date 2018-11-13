import os
import tensorflow as tf
import models.common.sequence as sequence
import models.common.vocab as vocab
import models.neural_editor.encoder as encoder
from models import neural_editor

from test.test_input_fn import embedding_file, dataset_file, dataset, EMBED_DIM, NUM_EPOCH, BATCH_SIZE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

HIDDEN_DIM = 256
NUM_LAYER = 3


def test_encoder(dataset_file, embedding_file):
    d_fn, gold_dataset = dataset_file
    e_fn, gold_embeds = embedding_file

    v, embed_matrix = vocab.read_word_embeddings(e_fn, EMBED_DIM)
    vocab_lookup = vocab.get_vocab_lookup(v)

    dataset = neural_editor.input_fn(d_fn, vocab_lookup, BATCH_SIZE, NUM_EPOCH)

    embedding = tf.get_variable('embeddings', shape=embed_matrix.shape,
                                initializer=tf.constant_initializer(embed_matrix))

    iter = dataset.make_initializable_iterator()
    (src, _, _, _), _ = iter.get_next()

    src_len = sequence.length_pre_embedding(src)
    src_embd = tf.nn.embedding_lookup(embedding, src)

    encoder_output, _ = encoder.bidirectional_encoder(src_embd, src_len, HIDDEN_DIM, NUM_LAYER, 0.9)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
        sess.run(iter.initializer)

        oeo, o_src, o_src_len, o_src_embd = sess.run([encoder_output, src, src_len, src_embd])

        for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print(i)

        assert oeo.shape == (BATCH_SIZE, o_src_len.max(), HIDDEN_DIM)
