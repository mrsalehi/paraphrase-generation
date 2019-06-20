import tensorflow as tf

import models.common.sequence as seq
from models.common.config import Config
from models.im_all_transformer import encoder
from models.im_all_transformer.decoder import Decoder
from models.im_all_transformer.edit_encoder import attn_encoder
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
        mev_st = mev_ts = None
    else:
        if draw_edit:
            edit_vector = random_noise_encoder(batch_size, config.editor.edit_encoder.edit_dim, config.editor.norm_max)
            mev_st = mev_ts = None
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

    beam_decoded_ids, beam_decoded_lengths, _ = decoder(None, None,
                                                        base_sent_hidden_states, base_sent_attention_bias,
                                                        edit_vector, mev_st, mev_ts,
                                                        mode='predict')

    return logits, beam_decoded_ids, beam_decoded_lengths
