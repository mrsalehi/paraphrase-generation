from typing import Tuple

import tensorflow as tf

import models.im_all_transformer.decoder as base_decoder
from models.common import sequence, vocab, graph_utils
from models.common.config import Config
from models.im_all_transformer.decoder import prepare_decoder_input
from models.im_all_transformer.transformer import model_utils
from models.im_all_transformer.transformer.embedding_layer import EmbeddingSharedWeights
from models.im_all_transformer_straight_attn import StraightAttentionDecoderStack

OPS_NAME = 'decoder'


class Decoder(base_decoder.Decoder):
    def __init__(self, config, **kwargs):
        config.put('editor.edit_encoder.edit_dim', 0)
        super().__init__(config, **kwargs)

        # Transformer stack
        del self.decoder_stack
        self.decoder_stack = StraightAttentionDecoderStack(self.config.to_json(), graph_utils.is_training())

    def _prepare_inputs(self, output_word_ids: tf.Tensor, edit_vector: tf.Tensor):
        # Add start token to decoder inputs
        decoder_input_words = prepare_decoder_input(output_word_ids)  # [batch, output_len+1]
        decoder_input_max_len = tf.shape(decoder_input_words)[1]
        decoder_input_len = sequence.length_pre_embedding(decoder_input_words)  # [batch]

        # Get word embeddings
        decoder_input_embeds = self.embedding_layer(decoder_input_words)  # [batch, output_len+1, hidden_size)

        # Add positional encoding to the embeddings part
        with tf.name_scope('positional_encoding'):
            pos_encoding = model_utils.get_position_encoding(decoder_input_max_len, self.config.orig_hidden_size)
            decoder_input_embeds += pos_encoding

        decoder_input = decoder_input_embeds

        if self.config.enable_dropout and self.config.layer_postprocess_dropout > 0.:
            decoder_input = tf.nn.dropout(decoder_input, 1 - self.config.layer_postprocess_dropout)

        return decoder_input, decoder_input_len

    def _decode_train(self, decoder_input: tf.Tensor, decoder_input_len: tf.Tensor,
                      base_sent_hidden_states: tf.Tensor, base_sent_attention_bias: tf.Tensor,
                      mev_st: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
                      mev_ts: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]):
        """Generate logits for each value in the target sequence.

        Returns:
          float32 tensor with shape [batch_size, output_length+1, vocab_size]
        """
        with tf.name_scope("decode_train"):
            # To prevent the model from looking in the future we need to mask
            # its self attention
            max_length = tf.shape(decoder_input)[1]
            decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(max_length)

            # Run transformer
            outputs = self.decoder_stack(
                decoder_input, decoder_self_attention_bias,
                encoder_outputs=base_sent_hidden_states, encoder_attn_bias=base_sent_attention_bias,
                mev_st=mev_st[0], mev_st_keys=mev_st[1], mev_st_attn_bias=mev_st[2],
                mev_ts=mev_ts[0], mev_ts_keys=mev_ts[1], mev_ts_attn_bias=mev_ts[2],
            )

            # Project transformer outputs to the embedding space
            outputs = self.project_back(outputs)
            logits = self.vocab_projection.linear(outputs)

            return logits

    def _get_symbols_to_logits_fn(self, edit_vector):
        """Returns a decoding function that calculates logits of the next tokens."""
        max_decode_length = self.config.max_decode_length

        timing_signal = model_utils.get_position_encoding(
            max_decode_length + 1,
            self.config.orig_hidden_size
        )
        decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(max_decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            """Generate logits for next potential IDs.

            Args:
              ids: Current decoded sequences.
                int tensor with shape [batch_size * beam_size, i + 1]
              i: Loop index
              cache: dictionary of values storing the encoder output, encoder-decoder
                attention bias, and previous decoder attention values.

            Returns:
              Tuple of
                (logits with shape [batch_size * beam_size, vocab_size],
                 updated cache values)
            """
            # Set decoder input to the last generated IDs
            decoder_input = ids[:, -1:]

            # Preprocess decoder input by getting embeddings and adding timing signal.
            decoder_input = self.embedding_layer(decoder_input)
            decoder_input += timing_signal[i:i + 1]

            self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

            outputs = self.decoder_stack(
                decoder_input, self_attention_bias,
                encoder_outputs=cache["encoder_outputs"], encoder_attn_bias=cache["encoder_attn_bias"],
                mev_st=cache["mev_st"], mev_st_keys=cache["mev_st_keys"], mev_st_attn_bias=cache["mev_st_attn_bias"],
                mev_ts=cache["mev_ts"], mev_ts_keys=cache["mev_ts_keys"], mev_ts_attn_bias=cache["mev_ts_attn_bias"],
                cache=cache
            )

            # Project transformer outputs to the embedding space
            outputs = self.project_back(outputs)
            logits = self.vocab_projection.linear(outputs)
            logits = tf.squeeze(logits, axis=[1])

            return logits, cache

        return symbols_to_logits_fn


def str_tokens(decoded_ids):
    vocab_i2s = vocab.get_vocab_lookup_tables()[vocab.INT_TO_STR]
    return vocab_i2s.lookup(
        tf.to_int64(decoded_ids)
    )


def logits_to_decoded_ids(logits):
    return tf.argmax(logits, axis=-1)
