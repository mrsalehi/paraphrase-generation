import tensorflow as tf

import models.common.sequence as seq
from models.common import vocab
from models.im_pointer_generator import decoder
from models.neural_editor import encoder
from models.neural_editor.decoder import prepare_decoder_inputs, prepare_decoder_output


def add_decoder_attn_history_graph(decoder_output):
    is_training = tf.get_collection('is_training')[0]
    if is_training:
        return

    alignment_history = decoder.alignment_history(decoder_output)
    alignment_history = list(map(lambda x: tf.transpose(x.stack(), [1, 0, 2]), alignment_history))
    tf.add_to_collection('decoder_alignment_history', alignment_history)


def prepare_decoder_input_output(words, words_extended, length):
    """
    Args:
        words: tensor of word ids, [batch x max_len]
        length: vector of sentence lengths, [batch]
        vocab_table: instance of tf.vocab_lookup_table

    Returns:
        dec_input: tensor of word ids, [batch x max_len+1]
        dec_input_len: vector of sentence lengths, [batch]
        dec_output: tensor of word ids, [batch x max_len+1]
        dec_output_len: vector of sentence lengths, [batch]

    """
    start_token_id = vocab.get_token_id(vocab.START_TOKEN)
    stop_token_id = vocab.get_token_id(vocab.STOP_TOKEN)
    pad_token_id = vocab.get_token_id(vocab.PAD_TOKEN)

    dec_input = prepare_decoder_inputs(words, start_token_id)
    dec_input_len = seq.length_pre_embedding(dec_input)

    dec_output = prepare_decoder_output(words_extended, length, stop_token_id, pad_token_id)
    dec_output_len = seq.length_pre_embedding(dec_output)

    return dec_input, dec_input_len, dec_output, dec_output_len


def linear(sentence_embed, agenda_dim, reuse=None):
    with tf.variable_scope('agenda_maker', 'agenda_maker', [sentence_embed], reuse=None):
        return tf.layers.dense(
            sentence_embed,
            agenda_dim
        )


def editor_train(base_words, extended_base_words, output_words, extended_output_words,
                 source_words, target_words, insert_words, delete_words, oov,
                 vocab_size, hidden_dim, agenda_dim, edit_dim, micro_edit_ev_dim, num_heads,
                 num_encoder_layers, num_decoder_layers, attn_dim, beam_width,
                 ctx_hidden_dim, ctx_hidden_layer, wa_hidden_dim, wa_hidden_layer, meve_hidden_dim, meve_hidden_layers,
                 max_sent_length, dropout_keep, lamb_reg, norm_eps, norm_max, kill_edit, draw_edit, swap_memory,
                 use_beam_decoder=False, use_dropout=False, no_insert_delete_attn=False, enable_vae=True):
    # [batch]
    base_len = seq.length_pre_embedding(base_words)
    output_len = seq.length_pre_embedding(extended_output_words)

    # variable of shape [vocab_size, embed_dim]
    embeddings = vocab.get_embeddings()

    # [batch x max_len x embed_dim]
    base_word_embeds = vocab.embed_tokens(base_words)

    # [batch x max_len x rnn_out_dim], [batch x rnn_out_dim]
    base_sent_hidden_states, base_sent_embed = encoder.source_sent_encoder(
        base_word_embeds,
        base_len,
        hidden_dim, num_encoder_layers,
        use_dropout=use_dropout, dropout_keep=dropout_keep, swap_memory=swap_memory
    )

    assert kill_edit == False and draw_edit == False

    # [batch x agenda_dim]
    base_agenda = linear(base_sent_embed, agenda_dim)

    train_dec_inp, train_dec_inp_len, \
    train_dec_out, train_dec_out_len = prepare_decoder_input_output(output_words, extended_output_words, output_len)

    train_dec_inp_extended = prepare_decoder_inputs(extended_output_words, tf.cast(-1, tf.int64))

    train_decoder = decoder.train_decoder(base_agenda, embeddings, extended_base_words, oov,
                                          train_dec_inp, train_dec_inp_extended, base_sent_hidden_states,
                                          train_dec_inp_len, base_len,
                                          vocab_size, attn_dim, hidden_dim, num_decoder_layers, swap_memory,
                                          enable_dropout=use_dropout, dropout_keep=dropout_keep,
                                          no_insert_delete_attn=no_insert_delete_attn)

    if use_beam_decoder:
        infr_decoder = decoder.beam_eval_decoder(base_agenda, embeddings, extended_base_words, oov,
                                                 vocab.get_token_id(vocab.START_TOKEN),
                                                 vocab.get_token_id(vocab.STOP_TOKEN),
                                                 base_sent_hidden_states, base_len,
                                                 vocab_size, attn_dim, hidden_dim,
                                                 num_decoder_layers, max_sent_length, beam_width,
                                                 swap_memory, enable_dropout=use_dropout, dropout_keep=dropout_keep,
                                                 no_insert_delete_attn=no_insert_delete_attn)
    else:
        infr_decoder = decoder.greedy_eval_decoder(base_agenda, embeddings, extended_base_words, oov,
                                                   vocab.get_token_id(vocab.START_TOKEN),
                                                   vocab.get_token_id(vocab.STOP_TOKEN),
                                                   base_sent_hidden_states, base_len,
                                                   vocab_size, attn_dim, hidden_dim,
                                                   num_decoder_layers, max_sent_length,
                                                   swap_memory, enable_dropout=use_dropout, dropout_keep=dropout_keep,
                                                   no_insert_delete_attn=no_insert_delete_attn)

        add_decoder_attn_history_graph(infr_decoder)

    return train_decoder, infr_decoder, train_dec_out, train_dec_out_len


def editor_test():
    pass
