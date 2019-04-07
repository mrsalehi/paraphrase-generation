import tensorflow as tf

OPS_NAME = 'se_agenda'


def concat(sentence_embed, edit_vector, agenda_dim, reuse=None):
    with tf.variable_scope(OPS_NAME, 'agenda_maker', [sentence_embed, edit_vector], reuse=None):
        if agenda_dim == sentence_embed.shape[1]:
            base_embed = sentence_embed
        else:
            base_embed = tf.layers.dense(sentence_embed, agenda_dim)

        return tf.concat([base_embed, edit_vector], 1)
