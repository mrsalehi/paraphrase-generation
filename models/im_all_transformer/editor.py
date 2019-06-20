import tensorflow as tf

import models.common.sequence as seq
# def editor_train(base_words, extended_base_words, output_words, extended_output_words,
#                  source_words, target_words, insert_words, delete_words, oov,
#                  vocab_size, hidden_dim, agenda_dim, edit_dim, micro_edit_ev_dim, num_heads,
#                  num_encoder_layers, num_decoder_layers, attn_dim, beam_width,
#                  transformer_params, wa_hidden_dim, wa_hidden_layer, meve_hidden_dim, meve_hidden_layers,
#                  max_sent_length, dropout_keep, lamb_reg, norm_eps, norm_max, kill_edit, draw_edit, swap_memory,
#                  use_beam_decoder=False, use_dropout=False, no_insert_delete_attn=False, enable_vae=True):
from models.common.config import Config
from models.im_all_transformer import encoder
from models.im_all_transformer.decoder import Decoder
from models.im_all_transformer.edit_encoder import attn_encoder
from models.im_all_transformer.transformer import model_utils
from models.im_all_transformer.transformer.embedding_layer import EmbeddingSharedWeights
from models.neural_editor.edit_encoder import random_noise_encoder


def editor_train(base_word_ids, output_word_ids,
                 source_word_ids, target_word_ids, insert_word_ids, common_word_ids,
                 config: Config):
    batch_size = tf.shape(base_word_ids)[0]

    # [batch]
    base_len = seq.length_pre_embedding(base_word_ids)
    output_len = seq.length_pre_embedding(output_word_ids)
    src_len = seq.length_pre_embedding(source_word_ids)
    tgt_len = seq.length_pre_embedding(target_word_ids)
    iw_len = seq.length_pre_embedding(insert_word_ids)
    cw_len = seq.length_pre_embedding(common_word_ids)

    embedding_layer = EmbeddingSharedWeights.get_from_graph()
    insert_embeds = embedding_layer(insert_word_ids)
    common_embeds = embedding_layer(common_word_ids)

    base_sent_hidden_states, base_sent_attention_bias = encoder.base_sent_encoder(base_word_ids, base_len, config)

    kill_edit = config.editor.kill_edit
    draw_edit = config.editor.draw_edit

    assert kill_edit == False and draw_edit == False

    if kill_edit:
        edit_vector = tf.zeros(shape=(batch_size, config.editor.edit_encoder.edit_dim))
    else:
        if draw_edit:
            edit_vector = random_noise_encoder(batch_size, config.editor.edit_encoder.edit_dim, config.editor.norm_max)
        else:
            edit_vector, mev_st, mev_ts = attn_encoder(
                source_word_ids, target_word_ids,
                insert_embeds, common_embeds,
                src_len, tgt_len, iw_len, cw_len,
                config
            )

    decoder = Decoder(config)
    logits = decoder(output_word_ids, output_len,
                     base_sent_hidden_states, base_sent_attention_bias,
                     edit_vector, mev_st, mev_ts,
                     mode='train')

    #
    # # [batch x agenda_dim]
    # base_agenda = agn.linear(base_sent_embed, edit_vector, agenda_dim)
    #
    # train_dec_inp, train_dec_inp_len, \
    # train_dec_out, train_dec_out_len = prepare_decoder_input_output(output_words, extended_output_words, output_len)
    #
    # train_dec_inp_extended = prepare_decoder_inputs(extended_output_words, tf.cast(-1, tf.int64))
    #
    # train_decoder = decoder.train_decoder(base_agenda, embeddings, extended_base_words, oov,
    #                                       train_dec_inp, train_dec_inp_extended, base_sent_hidden_states,
    #                                       wa_inserted, wa_deleted,
    #                                       train_dec_inp_len, base_len, src_len, tgt_len,
    #                                       vocab_size, attn_dim, hidden_dim, num_decoder_layers, swap_memory,
    #                                       enable_dropout=use_dropout, dropout_keep=dropout_keep,
    #                                       no_insert_delete_attn=no_insert_delete_attn)
    #
    # if use_beam_decoder:
    #     infr_decoder = decoder.beam_eval_decoder(base_agenda, embeddings, extended_base_words, oov,
    #                                              vocab.get_token_id(vocab.START_TOKEN),
    #                                              vocab.get_token_id(vocab.STOP_TOKEN),
    #                                              base_sent_hidden_states, wa_inserted, wa_deleted,
    #                                              base_len, src_len, tgt_len,
    #                                              vocab_size, attn_dim, hidden_dim,
    #                                              num_decoder_layers, max_sent_length, beam_width,
    #                                              swap_memory, enable_dropout=use_dropout, dropout_keep=dropout_keep,
    #                                              no_insert_delete_attn=no_insert_delete_attn)
    # else:
    #     infr_decoder = decoder.greedy_eval_decoder(base_agenda, embeddings, extended_base_words, oov,
    #                                                vocab.get_token_id(vocab.START_TOKEN),
    #                                                vocab.get_token_id(vocab.STOP_TOKEN),
    #                                                base_sent_hidden_states, wa_inserted, wa_deleted,
    #                                                base_len, src_len, tgt_len,
    #                                                vocab_size, attn_dim, hidden_dim,
    #                                                num_decoder_layers, max_sent_length,
    #                                                swap_memory, enable_dropout=use_dropout, dropout_keep=dropout_keep,
    #                                                no_insert_delete_attn=no_insert_delete_attn)
    #
    #     add_decoder_attn_history_graph(infr_decoder)
    #
    # return train_decoder, infr_decoder, train_dec_out, train_dec_out_len

    return logits
