import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
import numpy as np

import models.common.vocab as vocab
import models.common.sequence as seq

from models.neural_editor import encoder
from models.neural_editor import edit_encoder
from models.neural_editor import agenda

seq2seq.GreedyEmbeddingHelper
seq2seq.TrainingHelper
seq2seq.dynamic_decode()
seq2seq.AttentionWrapper
seq2seq.BasicDecoder




def editor_train(source_words, target_words, insert_words, delete_words, embed_matrix,
                 hidden_dim, agenda_dim, edit_dim, encoder_layer, decoder_layer,
                 ev_ctx_dim, ev_ctx_layer, ev_wa_dim, ev_wa_layer,
                 dropout_keep, lamb_reg, norm_eps, norm_max, kill_edit):
    # [batch]
    src_len = seq.length_pre_embedding(source_words)
    tgt_len = seq.length_pre_embedding(target_words)
    insert_words_len = seq.length_pre_embedding(insert_words)
    delete_words_len = seq.length_pre_embedding(delete_words)

    # variable of shape [vocab_size, embed_dim]
    embeddings = vocab.init_embeddings(embed_matrix)

    # [batch x max_len x embed_dim]
    source_words = vocab.embed_tokens(source_words)
    target_words = vocab.embed_tokens(target_words)
    insert_words = vocab.embed_tokens(insert_words)
    delete_words = vocab.embed_tokens(delete_words)

    source_embeds, src_sent_encoding = encoder.source_sent_encoder(
        source_words,
        src_len,
        hidden_dim, encoder_layer, dropout_keep
    )

    edit_vector = edit_encoder.accumulator_encoder(
        insert_words,
        delete_words,
        insert_words_len,
        delete_words_len,
        edit_dim, lamb_reg, norm_eps, norm_max, dropout_keep
    )

    src_agenda = agenda.linear(src_sent_encoding, edit_vector, agenda_dim)




def editor_test():
    pass
