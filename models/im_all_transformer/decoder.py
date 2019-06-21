from typing import Tuple

import tensorflow as tf
from tensorflow.contrib.seq2seq import FinalBeamSearchDecoderOutput

from models.common import sequence, vocab, graph_utils
from models.common.config import Config
from models.im_all_transformer.transformer import attention_layer, ffn_layer, model_utils, beam_search
from models.im_all_transformer.transformer.embedding_layer import EmbeddingSharedWeights
from models.im_all_transformer.transformer.transformer import PrePostProcessingWrapper, LayerNormalization

OPS_NAME = 'decoder'


def prepare_decoder_input(seq):
    start_token_id = vocab.get_token_id(vocab.START_TOKEN)

    batch_size = tf.shape(seq)[0]
    start_tokens = tf.fill([batch_size, 1], start_token_id)
    inputs = tf.concat([start_tokens, seq], axis=1)

    return inputs


def prepare_decoder_output(seq, seq_len):
    pad_token_id = vocab.get_token_id(vocab.PAD_TOKEN)
    stop_token_id = vocab.get_token_id(vocab.START_TOKEN)

    batch_size = tf.shape(seq)[0]

    extra_pad = tf.fill([batch_size, 1], pad_token_id)
    seq = tf.concat([seq, extra_pad], axis=1)
    max_length = tf.shape(seq)[1]

    update_indices = tf.range(0, batch_size) * max_length + (seq_len)
    update_indices = tf.reshape(update_indices, [-1, 1])
    flatten = tf.reshape(seq, [-1])

    updates = tf.fill([batch_size], stop_token_id)
    delta = tf.scatter_nd(update_indices, updates, tf.shape(flatten))

    outputs = flatten + delta
    outputs = tf.reshape(outputs, [-1, max_length])

    return outputs


class MultiSourceDecoderStack(tf.layers.Layer):
    """Transformer decoder stack.

    Like the encoder stack, the decoder stack is made up of N identical layers.
    Each layer is composed of the sublayers:
      1. Self-attention layer
      2. Multi-headed attention layer combining encoder outputs with results from
         the previous self-attention layer.
      3. Feedforward network (2 fully-connected layers)
    """

    def __init__(self, params, train):
        super(MultiSourceDecoderStack, self).__init__()
        self.params = params
        self.layers = []
        for _ in range(params["num_hidden_layers"]):
            self_attention_layer = attention_layer.SelfAttention(
                params["hidden_size"], params["num_heads"],
                params["attention_dropout"], train)

            hidden_layers = [PrePostProcessingWrapper(self_attention_layer, params, train)]

            enc_dec_attention_layer = attention_layer.Attention(
                params["hidden_size"], params["num_heads"],
                params["attention_dropout"], train)

            feed_forward_network = ffn_layer.FeedForwardNetwork(
                params["hidden_size"], params["filter_size"],
                params["relu_dropout"], train, params["allow_ffn_pad"])

            extra_attns = []
            if params['allow_mev_st_attn']:
                mev_st_attention_layer = attention_layer.Attention(
                    params["hidden_size"], params["num_heads"],
                    params["attention_dropout"], train)
                extra_attns.append(mev_st_attention_layer)

            if params['allow_mev_ts_attn']:
                mev_ts_attention_layer = attention_layer.Attention(
                    params["hidden_size"], params["num_heads"],
                    params["attention_dropout"], train)
                extra_attns.append(mev_ts_attention_layer)

            if len(extra_attns) > 0:
                attentions_layer_norm = LayerNormalization((len(extra_attns) + 1) * params["hidden_size"])
                hidden_layers += [
                    enc_dec_attention_layer,
                    feed_forward_network,
                    extra_attns,
                    attentions_layer_norm
                ]
            else:
                hidden_layers += [
                    PrePostProcessingWrapper(enc_dec_attention_layer, params, train),
                    PrePostProcessingWrapper(feed_forward_network, params, train)
                ]

            self.layers.append(hidden_layers)

        self.output_normalization = LayerNormalization(params["hidden_size"])

    def call(self, decoder_inputs, decoder_self_attention_bias, input_padding=None, cache=None, **kwargs):
        """Return the output of the decoder layer stacks.

        Args:
          decoder_inputs: tensor with shape [batch_size, target_length, hidden_size]
          decoder_self_attention_bias: bias for decoder self-attention layer.
            [1, 1, target_len, target_length]
          cache: (Used for fast decoding) A nested dictionary storing previous
            decoder self-attention values. The items are:
              {layer_n: {"k": tensor with shape [batch_size, i, key_channels],
                         "v": tensor with shape [batch_size, i, value_channels]},
               ...}

        Returns:
          Output of decoder layer stack.
          float32 tensor with shape [batch_size, target_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            feed_forward_network = layer[2]

            has_extra_attentions = len(layer) > 3

            # Run inputs through the sublayers.
            layer_name = "layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                with tf.variable_scope("self_attention"):
                    decoder_inputs = self_attention_layer(
                        decoder_inputs, decoder_self_attention_bias, cache=layer_cache)

                attns = self._compute_attentions(decoder_inputs, layer, kwargs)

                with tf.variable_scope("ffn"):
                    y = feed_forward_network(attns, input_padding)

                if has_extra_attentions:
                    if self.params["enable_dropout"] and self.params['layer_postprocess_dropout'] > 0.:
                        y = tf.nn.dropout(y, 1 - self.params['layer_postprocess_dropout'])

                    decoder_inputs = decoder_inputs + y
                else:
                    decoder_inputs = y

        return self.output_normalization(decoder_inputs)

    def _compute_attentions(self, decoder_inputs, layer, kwargs):
        encoder_outputs = kwargs['encoder_outputs']
        encoder_attention_bias = kwargs['encoder_attn_bias']
        enc_dec_attn_layer = layer[1]

        with tf.variable_scope("encdec_attention"):
            enc_dec_attn_result = enc_dec_attn_layer(
                decoder_inputs, encoder_outputs, encoder_attention_bias)

        if len(layer) == 3:
            return enc_dec_attn_result

        extra_attns = layer[3]
        results = []
        if len(extra_attns) > 0:
            with tf.variable_scope("mev_st_attention"):
                mev_st = kwargs['mev_st']
                mev_st_keys = kwargs['mev_st_keys']
                mev_st_attn_bias = kwargs['mev_st_attn_bias']
                mev_st_attention_layer = extra_attns[0]

                results.append(mev_st_attention_layer(
                    decoder_inputs, mev_st_keys, mev_st_attn_bias, z=mev_st))

        if len(extra_attns) == 2:
            with tf.variable_scope("mev_ts_attention"):
                mev_ts = kwargs['mev_ts']
                mev_ts_keys = kwargs['mev_ts_keys']
                mev_ts_attn_bias = kwargs['mev_ts_attn_bias']
                mev_ts_attention_layer = extra_attns[1]

                results.append(mev_ts_attention_layer(
                    decoder_inputs, mev_ts_keys, mev_ts_attn_bias, z=mev_ts))

        attentions_layer_norm = layer[4]
        attns = tf.concat([enc_dec_attn_result, *results], axis=-1)

        if self.params["enable_dropout"] and self.params['layer_postprocess_dropout'] > 0.:
            attns = tf.nn.dropout(attns, 1 - self.params['layer_postprocess_dropout'])

        attns = attentions_layer_norm(attns)

        return attns


class Decoder(tf.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = Config.merge_to_new([config.editor.transformer, config.editor.decoder])

        # change hidden_size of transformer to match with the augmented input
        self.config.put('orig_hidden_size', self.config.hidden_size)
        self.config.put('hidden_size', self.config.hidden_size + config.editor.edit_encoder.edit_dim)

        # Project embedding to transformer's hidden_size if needed
        embedding_layer = EmbeddingSharedWeights.get_from_graph()
        self.vocab_size = embedding_layer.vocab_size
        if config.editor.word_dim != self.config.orig_hidden_size:
            self.embedding_layer = embedding_layer.get_projected(self.config.orig_hidden_size)
        else:
            self.embedding_layer = embedding_layer

        # As far as EmbeddingSharedWeights class supports linear projection on embeddings,
        # we will use it to compute the model's logits
        self.project_back = tf.layers.Dense(config.editor.word_dim, activation=None, name='project_back')
        self.vocab_projection = embedding_layer

        # Transformer stack
        self.decoder_stack = MultiSourceDecoderStack(self.config.to_json(), graph_utils.is_training())

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

        # Add edit vector to each decoder stack
        # [batch, output_len+1, edit_dim]
        edit_vector = tf.tile(tf.expand_dims(edit_vector, axis=1), [1, decoder_input_max_len, 1])

        # [batch, output_len+1, hidden_size+edit_dim]
        decoder_input = tf.concat([decoder_input_embeds, edit_vector], axis=-1)

        if self.config.enable_dropout and self.config.layer_postprocess_dropout > 0.:
            decoder_input = tf.nn.dropout(decoder_input, 1 - self.config.layer_postprocess_dropout)

        return decoder_input, decoder_input_len

    def call(self, output_word_ids: tf.Tensor, output_lengths: tf.Tensor,
             base_sent_hidden_states: tf.Tensor, base_sent_attention_bias: tf.Tensor,
             edit_vector: tf.Tensor,
             mev_st: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
             mev_ts: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
             mode='train', **kwargs):

        """
        Args:
            output_word_ids: output sentence, shape = [batch, output_length]
            output_lengths: output sentence length, shape = [batch]
            base_sent_hidden_states: base sentence encoded via a Transformer encoder, shape = [batch, base_length, hidden_size]
            base_sent_attention_bias: base sentence mask, shape = [batch, 1, 1, base_length]
            edit_vector: edit vector extracted from <P,Q> pair, shape = [batch, edit_dim]
            mev_st: Micro edit vectors: P -> Q, tuple(
                        actual micro edit-vectors, shape=[batch, P_length, micro_edit_vector_dim],
                        P hidden_states. acts as a key for attention, shape=[batch, P_length, hidden_size],
                        attention_bias, shape=[batch, 1, 1, P_lengths]
                    )
            mev_ts: Micro edit vectors: Q -> P, tuple(
                        Q hidden_states. acts as a key for attention, shape=[batch, Q_length, hidden_size],
                        actual micro edit-vectors, shape=[batch, Q_length, micro_edit_vector_dim],
                        attention_bias, shape=[batch, 1, 1, P_lengths]
                    )
            mode: 'train' vs 'predict'

        Returns:
            Tensor with [batch, output_length+1, vocab_size] in mode == 'train'
        """
        if mode == 'train':
            decoder_input, decoder_input_len = self._prepare_inputs(output_word_ids, edit_vector)
            logits = self._decode_train(decoder_input, decoder_input_len,
                                        base_sent_hidden_states, base_sent_attention_bias,
                                        mev_st, mev_ts)

            return logits

        elif mode == 'predict':
            decoder_output = self._decode_predict(base_sent_hidden_states, base_sent_attention_bias,
                                                  edit_vector, mev_st, mev_ts)
            return decoder_output

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

    def _decode_predict(self, base_sent_hidden_states: tf.Tensor, base_sent_attention_bias: tf.Tensor,
                        edit_vector: tf.Tensor,
                        mev_st: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
                        mev_ts: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]):
        with tf.name_scope("decode_predict"):
            batch_size = tf.shape(base_sent_hidden_states)[0]

            symbols_to_logits_fn = self._get_symbols_to_logits_fn(edit_vector)

            # Create initial set of IDs that will be passed into symbols_to_logits_fn.
            start_id = tf.to_int32(vocab.get_token_id(vocab.START_TOKEN))
            initial_ids = tf.fill([batch_size], start_id)

            # Create cache storing decoder attention values for each layer.
            cache = {
                "layer_%d" % layer: {
                    "k": tf.zeros([batch_size, 0, self.config.hidden_size]),
                    "v": tf.zeros([batch_size, 0, self.config.hidden_size]),
                } for layer in range(self.config.num_hidden_layers)}

            # Add encoder output and attention bias to the cache.
            cache["encoder_outputs"] = base_sent_hidden_states
            cache["encoder_attn_bias"] = base_sent_attention_bias

            cache["mev_st"] = mev_st[0]
            cache['mev_st_keys'] = mev_st[1]
            cache['mev_st_attn_bias'] = mev_st[2]

            cache['mev_ts'] = mev_ts[0]
            cache['mev_ts_keys'] = mev_ts[1]
            cache['mev_ts_attn_bias'] = mev_ts[2]

            # Use beam search to find the top beam_size sequences and scores.
            decoded_ids, scores = beam_search.sequence_beam_search(
                symbols_to_logits_fn=symbols_to_logits_fn,
                initial_ids=initial_ids,
                initial_cache=cache,
                vocab_size=self.vocab_size,
                beam_size=self.config.beam_size,
                alpha=self.config.beam_decoding_alpha,
                max_decode_length=self.config.max_decode_length,
                eos_id=tf.to_int32(vocab.get_token_id(vocab.STOP_TOKEN)))

            # Change decoded_ids from [batch_size, beam_size, max_decode_length]
            # to [batch, max_decode_length, beam_size]
            decoded_ids = tf.transpose(decoded_ids, [0, 2, 1])
            decoded_ids = decoded_ids[:, 1:, :]

            # mask for valid tokens
            # [batch, max_decode_length, beam_size]
            mask = tf.to_int32(tf.not_equal(decoded_ids, 0))

            # [batch, beam_size]
            decoded_lengths = tf.reduce_sum(mask, axis=1)

            return decoded_ids, decoded_lengths, scores

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

            # Add edit_vector to decoder_input
            # [batch, 1, edit_vector_dim]
            expanded_edit_vector = tf.expand_dims(edit_vector, axis=1)
            expanded_edit_vector = tf.tile(expanded_edit_vector, [self.config.beam_size, 1, 1])

            # [batch, hidden_size = orig_hidden_size + edit_vector_dim]
            decoder_input = tf.concat([decoder_input, expanded_edit_vector], axis=-1)

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
