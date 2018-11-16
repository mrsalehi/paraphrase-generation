import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
import numpy as np

from models.common import vocab
import models.common.sequence as seq

from models.neural_editor import encoder
from models.neural_editor import edit_encoder
from models.neural_editor import agenda as agn
from models.neural_editor import decoder


def prepare_decoder_input_output(tgt_words, tgt_len, vocab_table):
    """
    Args:
        tgt_words: tensor of word ids, [batch x max_len]
        tgt_len: vector of sentence lengths, [batch]
        vocab_table: instance of tf.vocab_lookup_table

    Returns:
        dec_input: tensor of word ids, [batch x max_len+1]
        dec_input_len: vector of sentence lengths, [batch]
        dec_output: tensor of word ids, [batch x max_len+1]
        dec_output_len: vector of sentence lengths, [batch]

    """
    start_token_id = vocab.get_token_id(vocab.START_TOKEN, vocab_table)
    stop_token_id = vocab.get_token_id(vocab.STOP_TOKEN, vocab_table)
    pad_token_id = vocab.get_token_id(vocab.PAD_TOKEN, vocab_table)

    dec_input = decoder.prepare_decoder_inputs(tgt_words, start_token_id)
    dec_input_len = seq.length_pre_embedding(dec_input)

    dec_output = decoder.prepare_decoder_output(tgt_words, tgt_len, stop_token_id, pad_token_id)
    dec_output_len = seq.length_pre_embedding(dec_output)

    return dec_input, dec_input_len, dec_output, dec_output_len


def editor_train(source_words, target_words, insert_words, delete_words,
                 embed_matrix, vocab_table,
                 hidden_dim, agenda_dim, edit_dim, num_encoder_layers, num_decoder_layers, attn_dim,
                 max_sent_length, dropout_keep, lamb_reg, norm_eps, norm_max, kill_edit, draw_edit, swap_memory):
    batch_size = tf.shape(source_words)[0]

    # [batch]
    src_len = seq.length_pre_embedding(source_words)
    tgt_len = seq.length_pre_embedding(target_words)
    iw_len = seq.length_pre_embedding(insert_words)
    dw_len = seq.length_pre_embedding(delete_words)

    # variable of shape [vocab_size, embed_dim]
    embeddings = vocab.init_embeddings(embed_matrix)

    # [batch x max_len x embed_dim]
    src_word_embeds = vocab.embed_tokens(source_words)
    insert_word_embeds = vocab.embed_tokens(insert_words)
    delete_word_embeds = vocab.embed_tokens(delete_words)

    # [batch x max_len x rnn_out_dim], [batch x rnn_out_dim]
    src_sent_hidden_states, src_sent_embed = encoder.source_sent_encoder(
        src_word_embeds,
        src_len,
        hidden_dim, num_encoder_layers, dropout_keep
    )

    # [batch x edit_dim]
    if kill_edit:
        edit_vector = tf.zeros(shape=(batch_size, edit_dim))
    else:
        if draw_edit:
            edit_vector = edit_encoder.random_noise_encoder(batch_size, edit_dim, norm_max)
        else:
            edit_vector = edit_encoder.accumulator_encoder(
                insert_word_embeds,
                delete_word_embeds,
                iw_len,
                dw_len,
                edit_dim, lamb_reg, norm_eps, norm_max, dropout_keep
            )

    # [batch x agenda_dim]
    input_agenda = agn.linear(src_sent_embed, edit_vector, agenda_dim)

    train_dec_inp, train_dec_inp_len, \
    train_dec_out, train_dec_out_len = prepare_decoder_input_output(target_words, tgt_len, vocab_table)

    train_decoder = decoder.train_decoder(input_agenda, embeddings, train_dec_inp,
                                          src_sent_hidden_states, insert_word_embeds, delete_word_embeds,
                                          train_dec_inp_len, src_len, iw_len, dw_len,
                                          attn_dim, hidden_dim, num_decoder_layers, swap_memory)

    infr_decoder = decoder.greedy_eval_decoder(input_agenda, embeddings,
                                               vocab.get_token_id(vocab.START_TOKEN, vocab_table),
                                               vocab.get_token_id(vocab.STOP_TOKEN, vocab_table),
                                               src_sent_hidden_states, insert_word_embeds, delete_word_embeds,
                                               src_len, iw_len, dw_len,
                                               attn_dim, hidden_dim, num_decoder_layers, max_sent_length)

    return train_decoder, infr_decoder, train_dec_out, train_dec_out_len


def editor_test():
    pass
