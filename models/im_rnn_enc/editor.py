import tensorflow as tf

import models.common.sequence as seq
from models.common import vocab
from models.im_rnn_enc.edit_encoder import rnn_encoder
from models.neural_editor import agenda as agn
from models.neural_editor import decoder
from models.neural_editor import encoder
from models.neural_editor.edit_encoder import random_noise_encoder
from models.neural_editor.editor import prepare_decoder_input_output


def editor_train(base_words, source_words, target_words, insert_words, delete_words,
                 hidden_dim, agenda_dim, edit_dim, num_encoder_layers, num_decoder_layers, attn_dim, beam_width,
                 ctx_hidden_dim, ctx_hidden_layer, wa_hidden_dim, wa_hidden_layer,
                 max_sent_length, dropout_keep, lamb_reg, norm_eps, norm_max, kill_edit, draw_edit, swap_memory,
                 use_beam_decoder=False, use_dropout=False):
    batch_size = tf.shape(source_words)[0]

    # [batch]
    base_len = seq.length_pre_embedding(base_words)
    src_len = seq.length_pre_embedding(source_words)
    tgt_len = seq.length_pre_embedding(target_words)
    iw_len = seq.length_pre_embedding(insert_words)
    dw_len = seq.length_pre_embedding(delete_words)

    # variable of shape [vocab_size, embed_dim]
    embeddings = vocab.get_embeddings()

    # [batch x max_len x embed_dim]
    base_word_embeds = vocab.embed_tokens(base_words)
    src_word_embeds = vocab.embed_tokens(source_words)
    tgt_word_embeds = vocab.embed_tokens(target_words)
    insert_word_embeds = vocab.embed_tokens(insert_words)
    delete_word_embeds = vocab.embed_tokens(delete_words)

    # [batch x max_len x rnn_out_dim], [batch x rnn_out_dim]
    base_sent_hidden_states, base_sent_embed = encoder.source_sent_encoder(
        base_word_embeds,
        base_len,
        hidden_dim, num_encoder_layers,
        use_dropout=use_dropout, dropout_keep=dropout_keep, swap_memory=swap_memory
    )

    # [batch x edit_dim]
    if kill_edit:
        edit_vector = tf.zeros(shape=(batch_size, edit_dim))
    else:
        if draw_edit:
            edit_vector = random_noise_encoder(batch_size, edit_dim, norm_max)
        else:
            edit_vector = rnn_encoder(
                src_word_embeds, tgt_word_embeds,
                insert_word_embeds, delete_word_embeds,
                src_len, tgt_len,
                iw_len, dw_len,
                ctx_hidden_dim, ctx_hidden_layer, wa_hidden_dim, wa_hidden_layer,
                edit_dim, lamb_reg, norm_eps, norm_max,
                use_dropout=use_dropout, dropout_keep=dropout_keep, swap_memory=swap_memory
            )

    # [batch x agenda_dim]
    input_agenda = agn.linear(base_sent_embed, edit_vector, agenda_dim)

    train_dec_inp, train_dec_inp_len, \
    train_dec_out, train_dec_out_len = prepare_decoder_input_output(target_words, tgt_len, None)

    train_decoder = decoder.train_decoder(input_agenda, embeddings, train_dec_inp,
                                          base_sent_hidden_states, insert_word_embeds, delete_word_embeds,
                                          train_dec_inp_len, src_len, iw_len, dw_len,
                                          attn_dim, hidden_dim, num_decoder_layers, swap_memory,
                                          enable_dropout=use_dropout, dropout_keep=dropout_keep)

    if use_beam_decoder:
        infr_decoder = decoder.beam_eval_decoder(input_agenda, embeddings,
                                                 vocab.get_token_id(vocab.START_TOKEN),
                                                 vocab.get_token_id(vocab.STOP_TOKEN),
                                                 base_sent_hidden_states, insert_word_embeds, delete_word_embeds,
                                                 src_len, iw_len, dw_len,
                                                 attn_dim, hidden_dim, num_decoder_layers, max_sent_length, beam_width,
                                                 swap_memory, enable_dropout=use_dropout, dropout_keep=dropout_keep)
    else:
        infr_decoder = decoder.greedy_eval_decoder(input_agenda, embeddings,
                                                   vocab.get_token_id(vocab.START_TOKEN),
                                                   vocab.get_token_id(vocab.STOP_TOKEN),
                                                   base_sent_hidden_states, insert_word_embeds, delete_word_embeds,
                                                   src_len, iw_len, dw_len,
                                                   attn_dim, hidden_dim, num_decoder_layers, max_sent_length,
                                                   swap_memory, enable_dropout=use_dropout, dropout_keep=dropout_keep)

    return train_decoder, infr_decoder, train_dec_out, train_dec_out_len


def editor_test():
    pass
