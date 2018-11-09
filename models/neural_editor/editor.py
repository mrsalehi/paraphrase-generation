import tensorflow as tf
import numpy as np


def editor_train(src, tgt, insert_words, delete_words, embedding,
                 embed_matrix, hidden_dim, agenda_dim, edit_dim, encoder_layer, decoder_layer,
                 ev_ctx_dim, ev_ctx_layer, ev_wa_dim, ev_wa_layer,
                 dropout, lamb_reg, norm_eps, norm_max, kill_edit):
    tf.nn.bidirectional_dynamic_rnn()
    pass


def editor_test():
    pass
