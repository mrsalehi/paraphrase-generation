import tensorflow as tf
import tensorflow.contrib.rnn as tf_rnn
from tensorflow.contrib import seq2seq
from tensorflow_probability import distributions as tfd

from models.common import vocab
from models.common.sequence import create_trainable_initial_states
from models.im_attn_ee_rnn_attn_dec_pg import decoder
from models.neural_editor import encoder
from models.neural_editor.decoder import DecoderOutputLayer
from models.neural_editor.editor import prepare_decoder_input_output

OPS_NAME = 'reconstruction'


def prepare_output_embed(decoder_output, temperature_starter, decay_rate, decay_steps, ):
    # [VOCAB x word_dim]
    embeddings = vocab.get_embeddings()

    # Extend embedding matrix to support oov tokens
    unk_id = vocab.get_token_id(vocab.UNKNOWN_TOKEN)
    unk_embed = tf.expand_dims(vocab.embed_tokens(unk_id), 0)
    unk_embeddings = tf.tile(unk_embed, [50, 1])

    # [VOCAB+50 x word_dim]
    embeddings_extended = tf.concat([embeddings, unk_embeddings], axis=0)

    global_step = tf.train.get_global_step()
    temperature = tf.train.exponential_decay(
        temperature_starter,
        global_step,
        decay_steps, decay_rate,
        name='temperature'
    )
    tf.summary.scalar('temper', temperature, ['extra'])

    # [batch x max_len x VOCAB+50], softmax probabilities
    outputs = decoder.rnn_output(decoder_output)

    # substitute values less than 0 for numerical stability
    outputs = tf.where(tf.less_equal(outputs, 0), tf.ones_like(outputs) * 1e-10, outputs)

    # convert softmax probabilities to one_hot vectors
    dist = tfd.RelaxedOneHotCategorical(temperature, probs=outputs)

    # [batch x max_len x VOCAB+50], one_hot
    outputs_one_hot = dist.sample()

    # [batch x max_len x word_dim], one_hot^T * embedding_matrix
    outputs_embed = tf.einsum("btv,vd-> btd", outputs_one_hot, embeddings_extended)

    return outputs_embed


def create_rnn_layer(layer_num, dim, use_dropout, dropout_keep):
    cell = tf_rnn.LSTMCell(dim, name='layer_%s' % layer_num)
    if use_dropout and dropout_keep < 1.:
        cell = tf_rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep)

    if layer_num > 0:
        cell = tf_rnn.ResidualWrapper(cell)

    return cell


def residual_decoder(agenda, dec_inputs, dec_input_lengths,
                     hidden_dim, num_layer,
                     swap_memory, enable_dropout=False, dropout_keep=1., name=None):
    with tf.variable_scope(name, 'residual_decoder', []):
        batch_size = tf.shape(dec_inputs)[0]
        embeddings = vocab.get_embeddings()

        # Concatenate agenda [y_hat;base_input_embed] with decoder inputs

        # [batch x max_len x word_dim]
        dec_inputs = tf.nn.embedding_lookup(embeddings, dec_inputs)
        max_len = tf.shape(dec_inputs)[1]

        # [batch x 1 x agenda_dim]
        agenda = tf.expand_dims(agenda, axis=1)

        # [batch x max_len x agenda_dim]
        agenda = tf.tile(agenda, [1, max_len, 1])

        # [batch x max_len x word_dim+agenda_dim]
        dec_inputs = tf.concat([dec_inputs, agenda], axis=2)

        helper = seq2seq.TrainingHelper(dec_inputs, dec_input_lengths, name='train_helper')
        cell = tf_rnn.MultiRNNCell(
            [create_rnn_layer(i, hidden_dim // 2, enable_dropout, dropout_keep)
             for i in range(num_layer)]
        )
        zero_states = create_trainable_initial_states(batch_size, cell)

        output_layer = DecoderOutputLayer(embeddings)
        decoder = seq2seq.BasicDecoder(cell, helper, zero_states, output_layer)

        outputs, state, length = seq2seq.dynamic_decode(decoder, swap_memory=swap_memory)

        return outputs, state, length


def decode_agenda(agenda, p_dec_inp, q_dec_inp, p_dec_inp_len, q_dec_inp_len, dec_hidden_dim, dec_num_layer,
                  swap_memory, use_dropout=False, dropout_keep=1.0):
    p_dec = residual_decoder(agenda, p_dec_inp, p_dec_inp_len,
                             dec_hidden_dim, dec_num_layer, swap_memory,
                             enable_dropout=use_dropout, dropout_keep=dropout_keep, name='p_dec')

    q_dec = residual_decoder(agenda, q_dec_inp, q_dec_inp_len,
                             dec_hidden_dim, dec_num_layer, swap_memory,
                             enable_dropout=use_dropout, dropout_keep=dropout_keep, name='q_dec')

    return p_dec, q_dec


def decoding_loss(output, gold_rnn_output, lengths):
    # [batch x max_len x VOCAB_SIZE]
    rnn_output = decoder.rnn_output(output)

    # [batch x max_len x VOCAB_SIZE]
    mask = tf.sequence_mask(lengths, dtype=tf.float32)

    # [batch x max_len]
    loss = seq2seq.sequence_loss(
        rnn_output,
        gold_rnn_output,
        weights=mask,
        average_across_timesteps=False,
        average_across_batch=False
    )

    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))

    return loss


def decoder_outputs_to_pq(decoder_output, base_sent_embed, src_words, tgt_words, src_len, tgt_len,
                          temperature_starter, decay_rate, decay_steps,
                          agenda_dim, enc_hidden_dim, enc_num_layers, dec_hidden_dim, dec_num_layers,
                          swap_memory, use_dropout=False, dropout_keep=1.0):
    with tf.variable_scope(OPS_NAME):
        # [batch]
        output_length = decoder.seq_length(decoder_output)

        # [batch x max_len x word_dim]
        output_embed = prepare_output_embed(decoder_output, temperature_starter, decay_rate, decay_steps)

        # [batch x max_len x hidden], [batch x hidden]
        hidden_states, dec_output_embedding = encoder.bidirectional_encoder(
            output_embed,
            output_length,
            enc_hidden_dim, enc_num_layers,
            use_dropout=False, dropout_keep=1.0, swap_memory=swap_memory, name='dec_output_encoder'
        )

        agenda = tf.concat([dec_output_embedding, base_sent_embed], axis=1)
        agenda = tf.layers.dense(agenda, agenda_dim, activation=None, use_bias=False,
                                 name='agenda')  # [batch x agenda_dim]

        # append START, STOP token to create decoder input and output
        out = prepare_decoder_input_output(src_words, src_len, None)
        p_dec_inp, p_dec_inp_len, p_dec_out, p_dec_out_len = out

        out = prepare_decoder_input_output(tgt_words, tgt_len, None)
        q_dec_inp, q_dec_inp_len, q_dec_out, q_dec_out_len = out

        # decode agenda twice
        p_dec, q_dec = decode_agenda(agenda, p_dec_inp, q_dec_inp, p_dec_inp_len, q_dec_inp_len,
                                     dec_hidden_dim, dec_num_layers, swap_memory,
                                     use_dropout=use_dropout, dropout_keep=dropout_keep)

        # calculate the loss
        with tf.name_scope('losses'):
            p_loss = decoding_loss(p_dec, p_dec_out, p_dec_out_len)
            q_loss = decoding_loss(q_dec, q_dec_out, q_dec_out_len)

            loss = p_loss + q_loss

    tf.summary.scalar('reconstruction_loss', loss, ['extra'])

    return loss
