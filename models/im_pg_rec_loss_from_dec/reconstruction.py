import tensorflow as tf
from tensorflow_probability import distributions as tfd

from models.common import vocab
from models.im_attn_ee_rnn_attn_dec_pg import decoder
from models.neural_editor import encoder

OPS_NAME = 'reconstruction'


def decoder_outputs_to_edit_vector(decoder_output, temperature_starter, decay_rate, decay_steps,
                                   edit_dim, enc_hidden_dim, enc_num_layers, dense_layers,
                                   swap_memory):
    with tf.variable_scope(OPS_NAME):
        embeddings = vocab.get_embeddings()

        unk_id = vocab.get_token_id(vocab.UNKNOWN_TOKEN)
        unk_embed = tf.expand_dims(vocab.embed_tokens(unk_id), 0)
        unk_embeddings = tf.tile(unk_embed, [50, 1])
        embeddings_extended = tf.concat([embeddings, unk_embeddings], axis=0)

        global_step = tf.train.get_global_step()
        temperature = tf.train.exponential_decay(
            temperature_starter,
            global_step,
            decay_steps, decay_rate,
            name='temperature'
        )
        tf.summary.scalar('temperature', temperature, ['extra'])

        outputs = decoder.rnn_output(decoder_output)  # [b x t x V]
        outputs = tf.where(tf.less_equal(outputs, 0), tf.ones_like(outputs) * 1e-10, outputs)
        dist = tfd.RelaxedOneHotCategorical(temperature, probs=outputs)
        outputs_one_hot = dist.sample()

        outputs_embed = tf.einsum("btv,vd-> btd", outputs_one_hot, embeddings_extended)
        lengths = decoder.seq_length(decoder_output)

        hidden_states, seq_embedding = encoder.source_sent_encoder(
            outputs_embed,
            lengths,
            enc_hidden_dim, enc_num_layers,
            use_dropout=False, dropout_keep=1.0, swap_memory=swap_memory
        )

        h = seq_embedding
        for l in dense_layers:
            h = tf.layers.dense(h, l, activation='relu', name='hidden_%s' % (l))

        edit_vector = tf.layers.dense(h, edit_dim, activation=None, name='linear')

    return edit_vector
