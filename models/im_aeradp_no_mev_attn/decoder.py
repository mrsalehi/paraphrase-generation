import tensorflow as tf
import tensorflow.contrib.rnn as tf_rnn
from tensorflow.contrib import seq2seq

from models.common import vocab
from models.im_attn_ee_rnn_attn_dec_pg.decoder import create_trainable_zero_state, PointerGeneratorWrapper, \
    DecoderOutputLayer, AttentionAugmentRNNCell

OPS_NAME = 'decoder'


def create_decoder_cell(agenda, extended_base_words, oov, base_sent_hiddens, mev_st, mev_ts,
                        base_length, iw_length, dw_length,
                        vocab_size, attn_dim, hidden_dim, num_layer,
                        enable_alignment_history=False, enable_dropout=False, dropout_keep=1.,
                        no_insert_delete_attn=False, beam_width=None):
    base_attn = seq2seq.BahdanauAttention(attn_dim, base_sent_hiddens, base_length, name='base_attn')

    # cnx_src, micro_evs_st = mev_st
    # mev_st_attn = seq2seq.BahdanauAttention(attn_dim, cnx_src, iw_length, name='mev_st_attn')
    # mev_st_attn._values = micro_evs_st

    attns = [base_attn]

    if not no_insert_delete_attn:
        cnx_tgt, micro_evs_ts = mev_ts
        mev_ts_attn = seq2seq.BahdanauAttention(attn_dim, cnx_tgt, dw_length, name='mev_ts_attn')
        mev_ts_attn._values = micro_evs_ts

        attns += [mev_ts_attn]

    bottom_cell = tf_rnn.LSTMCell(hidden_dim, name='bottom_cell')
    bottom_attn_cell = seq2seq.AttentionWrapper(
        bottom_cell,
        tuple(attns),
        alignment_history=enable_alignment_history,
        output_attention=False,
        name='att_bottom_cell'
    )

    all_cells = [bottom_attn_cell]

    num_layer -= 1
    for i in range(num_layer):
        cell = tf_rnn.LSTMCell(hidden_dim, name='layer_%s' % (i + 1))
        if enable_dropout and dropout_keep < 1.:
            cell = tf_rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep)

        all_cells.append(cell)

    decoder_cell = AttentionAugmentRNNCell(all_cells)
    decoder_cell.set_agenda(agenda)
    decoder_cell.set_source_attn_index(0)

    output_layer = DecoderOutputLayer(vocab.get_embeddings())

    pg_cell = PointerGeneratorWrapper(
        decoder_cell,
        extended_base_words,
        50,
        output_layer,
        vocab_size,
        decoder_cell.get_source_attention,
        name='PointerGeneratorWrapper'
    )

    if beam_width:
        true_batch_size = tf.cast(tf.shape(base_sent_hiddens)[0] / beam_width, tf.int32)
    else:
        true_batch_size = tf.shape(base_sent_hiddens)[0]

    zero_state = create_trainable_zero_state(decoder_cell, true_batch_size, beam_width=beam_width)

    return pg_cell, zero_state
