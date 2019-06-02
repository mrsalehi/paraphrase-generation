import tensorflow as tf

from models.im_aeradp_rec_loss_from_dec_to_pq.reconstruction import decode_agenda, decoding_loss
from models.neural_editor import decoder
from models.neural_editor.editor import prepare_decoder_input_output

OPS_NAME = 'reconstruction'


def decoder_hiddens_to_pq(decoder_output, base_sent_embed, src_words, tgt_words, src_len, tgt_len,
                          agenda_dim, dec_hidden_dim, dec_num_layers,
                          swap_memory, use_dropout=False, dropout_keep=1.0):
    with tf.variable_scope(OPS_NAME):
        dec_output_embedding = decoder.last_hidden_state(decoder_output)
        dec_output_embedding = dec_output_embedding[-1].h

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
