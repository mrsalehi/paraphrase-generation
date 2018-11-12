import tensorflow as tf

OPS_NAME = 'se_agenda'


def linear(sentence_embed, edit_vector, agenda_dim, reuse=None):
    with tf.variable_scope(OPS_NAME, 'agenda_maker', [sentence_embed, edit_vector], reuse=None):
        return tf.layers.dense(
            tf.concat([sentence_embed, edit_vector], 1),
            agenda_dim
        )
