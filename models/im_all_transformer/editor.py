import tensorflow as tf

import models.common.sequence as seq
from models.common.config import Config
from models.im_all_transformer.decoder import Decoder
from models.im_all_transformer.edit_encoder import EditEncoder
from models.im_all_transformer.encoder import TransformerEncoder
from models.neural_editor.edit_encoder import random_noise_encoder


class Editor:
    def __init__(self, config: Config):
        self.config = config

        encoder_config = Config.merge_to_new([config.editor.transformer, config.editor.encoder])
        self.encoder = TransformerEncoder(encoder_config, name='encoder')
        self.edit_encoder = EditEncoder(config)
        self.decoder = Decoder(config)

    def encode_all(self, base_word_ids,
                   source_word_ids, target_word_ids,
                   insert_word_ids, common_word_ids):
        batch_size = tf.shape(base_word_ids)[0]

        with tf.name_scope('encode_all'):
            base_len = seq.length_pre_embedding(base_word_ids)
            src_len = seq.length_pre_embedding(source_word_ids)
            tgt_len = seq.length_pre_embedding(target_word_ids)
            iw_len = seq.length_pre_embedding(insert_word_ids)
            cw_len = seq.length_pre_embedding(common_word_ids)

            base_encoded, base_attention_bias = self.encoder(base_word_ids, base_len)

            kill_edit = self.config.editor.kill_edit
            draw_edit = self.config.editor.draw_edit

            if self.config.editor.decoder.allow_mev_st_attn \
                    or self.config.editor.allow_mev_ts_attn:
                assert kill_edit == False and draw_edit == False

            if kill_edit:
                edit_vector = tf.zeros(shape=(batch_size, self.config.editor.edit_encoder.edit_dim))
                mev_st = mev_ts = None
            else:
                if draw_edit:
                    edit_vector = random_noise_encoder(
                        batch_size,
                        self.config.editor.edit_encoder.edit_dim,
                        self.config.editor.norm_max)
                    mev_st = mev_ts = None
                else:
                    edit_vector, mev_st, mev_ts = self.edit_encoder(
                        source_word_ids, target_word_ids,
                        insert_word_ids, common_word_ids,
                        src_len, tgt_len, iw_len, cw_len,
                    )

            encoder_outputs = (base_encoded, base_attention_bias)
            edit_encoder_outputs = (edit_vector, mev_st, mev_ts)

            return encoder_outputs, edit_encoder_outputs

    def get_logits(self, encoded_inputs, output_word_ids):
        with tf.name_scope('logits'):
            encoder_outputs, edit_encoder_outputs = encoded_inputs

            base_sent_hidden_states, base_sent_attention_bias = encoder_outputs
            edit_vector, mev_st, mev_ts = edit_encoder_outputs

            output_len = seq.length_pre_embedding(output_word_ids)
            logits = self.decoder(output_word_ids, output_len,
                                  base_sent_hidden_states, base_sent_attention_bias,
                                  edit_vector, mev_st, mev_ts,
                                  mode='train')

            return logits

    def beam_predict(self, encoded_inputs):
        with tf.name_scope('beam_predict'):
            encoder_outputs, edit_encoder_outputs = encoded_inputs

            base_sent_hidden_states, base_sent_attention_bias = encoder_outputs
            edit_vector, mev_st, mev_ts = edit_encoder_outputs

            prediction = self.decoder(
                None, None,
                base_sent_hidden_states, base_sent_attention_bias,
                edit_vector, mev_st, mev_ts,
                mode='predict'
            )

            return prediction

    def __call__(self, base_word_ids, source_word_ids, target_word_ids,
                 insert_word_ids, common_word_ids, output_word_ids, **kwargs):
        initializer = tf.variance_scaling_initializer(
            self.config.editor.initializer_gain,
            mode="fan_avg",
            distribution="uniform"
        )

        with tf.variable_scope("editor", initializer=initializer):
            encoded_inputs = self.encode_all(
                base_word_ids,
                source_word_ids, target_word_ids,
                insert_word_ids, common_word_ids
            )

            logits = self.get_logits(encoded_inputs, output_word_ids)
            predictions = self.beam_predict(encoded_inputs)

            return (logits, predictions)

# def editor_train(base_word_ids, output_word_ids,
#                  source_word_ids, target_word_ids, insert_word_ids, common_word_ids,
#                  config: Config):
#     batch_size = tf.shape(base_word_ids)[0]
#
#     # [batch]
#     base_len = seq.length_pre_embedding(base_word_ids)
#     output_len = seq.length_pre_embedding(output_word_ids)
#     src_len = seq.length_pre_embedding(source_word_ids)
#     tgt_len = seq.length_pre_embedding(target_word_ids)
#     iw_len = seq.length_pre_embedding(insert_word_ids)
#     cw_len = seq.length_pre_embedding(common_word_ids)
#
#     embedding_layer = EmbeddingSharedWeights.get_from_graph()
#     insert_embeds = embedding_layer(insert_word_ids)
#     common_embeds = embedding_layer(common_word_ids)
#
#     base_sent_hidden_states, base_sent_attention_bias = encoder.base_sent_encoder(base_word_ids, base_len, config)
#
#     kill_edit = config.editor.kill_edit
#     draw_edit = config.editor.draw_edit
#
#     assert kill_edit == False and draw_edit == False
#
#     if kill_edit:
#         edit_vector = tf.zeros(shape=(batch_size, config.editor.edit_encoder.edit_dim))
#         mev_st = mev_ts = None
#     else:
#         if draw_edit:
#             edit_vector = random_noise_encoder(batch_size, config.editor.edit_encoder.edit_dim, config.editor.norm_max)
#             mev_st = mev_ts = None
#         else:
#             edit_vector, mev_st, mev_ts = attn_encoder(
#                 source_word_ids, target_word_ids,
#                 insert_embeds, common_embeds,
#                 src_len, tgt_len, iw_len, cw_len,
#                 config
#             )
#
#     decoder = Decoder(config)
#     logits = decoder(output_word_ids, output_len,
#                      base_sent_hidden_states, base_sent_attention_bias,
#                      edit_vector, mev_st, mev_ts,
#                      mode='train')
#
#     beam_decoded_ids, beam_decoded_lengths, _ = decoder(None, None,
#                                                         base_sent_hidden_states, base_sent_attention_bias,
#                                                         edit_vector, mev_st, mev_ts,
#                                                         mode='predict')
#
#     return logits, beam_decoded_ids, beam_decoded_lengths
