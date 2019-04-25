import tensorflow.contrib.rnn as tf_rnn
from tensorflow.contrib import seq2seq

from models.neural_editor.decoder import AttentionAugmentRNNCell


def create_decoder_cell(agenda, base_sent_embeds, mev_st, mev_ts,
                        base_length, iw_length, dw_length,
                        attn_dim, hidden_dim, num_layer,
                        enable_alignment_history=False, enable_dropout=False, dropout_keep=0.1,
                        no_insert_delete_attn=False):
    base_attn = seq2seq.BahdanauAttention(attn_dim, base_sent_embeds, base_length, name='src_attn')

    cnx_src, micro_evs_st = mev_st
    mev_st_attn = seq2seq.BahdanauAttention(attn_dim, cnx_src, iw_length, name='mev_st_attn')
    mev_st_attn._values = micro_evs_st

    attns = [base_attn, mev_st_attn]

    if not no_insert_delete_attn:
        cnx_tgt, micro_evs_ts = mev_ts
        mev_ts_attn = seq2seq.BahdanauAttention(attn_dim, cnx_tgt, dw_length, name='mev_ts_attn')
        mev_ts_attn._values = micro_evs_ts

        attns += [mev_ts_attn]

    bottom_cell = tf_rnn.LSTMCell(hidden_dim, name='bottom_cell')
    bottom_attn_cell = seq2seq.AttentionWrapper(
        bottom_cell,
        tuple(attns),
        output_attention=False,
        alignment_history=enable_alignment_history,
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

    return decoder_cell
