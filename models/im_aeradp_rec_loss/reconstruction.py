import tensorflow as tf

from models.neural_editor import encoder

OPS_NAME = 'reconstruction'


def output_words_to_edit_vector(output_words_embed, output_words_len,
                                edit_dim, enc_hidden_dim, enc_num_layers, dense_layers,
                                swap_memory):
    with tf.variable_scope(OPS_NAME):
        hidden_states, seq_embedding = encoder.source_sent_encoder(
            output_words_embed,
            output_words_len,
            enc_hidden_dim, enc_num_layers,
            use_dropout=False, dropout_keep=1.0, swap_memory=swap_memory
        )

        h = seq_embedding
        for l in dense_layers:
            h = tf.layers.dense(h, l, activation='relu', name='hidden_%s' % (l))

        edit_vector = tf.layers.dense(h, edit_dim, activation=None, name='linear')

    return edit_vector
