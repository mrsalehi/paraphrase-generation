import tensorflow as tf

from models.im_all_transformer.transformer import attention_layer, ffn_layer
from models.im_all_transformer.transformer.transformer import PrePostProcessingWrapper, LayerNormalization


class StraightAttentionDecoderStack(tf.layers.Layer):
    """Transformer decoder stack.

    Like the encoder stack, the decoder stack is made up of N identical layers.
    Each layer is composed of the sublayers:
      1. Self-attention layer
      2. Multi-headed attention layer combining encoder outputs with results from
         the previous self-attention layer.
      3. Feedforward network (2 fully-connected layers)
    """

    def __init__(self, params, train):
        super(StraightAttentionDecoderStack, self).__init__()
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
            hidden_layers += [PrePostProcessingWrapper(enc_dec_attention_layer, params, train), ]

            feed_forward_network = ffn_layer.FeedForwardNetwork(
                params["hidden_size"], params["filter_size"],
                params["relu_dropout"], train, params["allow_ffn_pad"])
            hidden_layers += [PrePostProcessingWrapper(feed_forward_network, params, train)]

            if params['allow_mev_st_attn']:
                mev_st_attention_layer = attention_layer.Attention(
                    params["hidden_size"], params["num_heads"],
                    params["attention_dropout"], train)
                hidden_layers += [PrePostProcessingWrapper(mev_st_attention_layer, params, train)]

            if params['allow_mev_ts_attn']:
                mev_ts_attention_layer = attention_layer.Attention(
                    params["hidden_size"], params["num_heads"],
                    params["attention_dropout"], train)
                hidden_layers += [PrePostProcessingWrapper(mev_ts_attention_layer, params, train)]

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
            enc_dec_attn_layer = layer[1]
            feed_forward_network = layer[2]

            # Run inputs through the sublayers.
            layer_name = "layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                with tf.variable_scope("self_attention"):
                    decoder_inputs = self_attention_layer(
                        decoder_inputs, decoder_self_attention_bias, cache=layer_cache)

                with tf.variable_scope("encdec_attention"):
                    decoder_inputs = enc_dec_attn_layer(
                        decoder_inputs, kwargs['encoder_outputs'], kwargs['encoder_attn_bias'])

                if len(layer) >= 4:
                    mev_st_attention_layer = layer[3]
                    with tf.variable_scope("mev_st_attention"):
                        decoder_inputs = mev_st_attention_layer(
                            decoder_inputs, kwargs['mev_st_keys'], kwargs['mev_st_attn_bias'], z=kwargs['mev_st'])

                if len(layer) >= 5:
                    mev_ts_attention_layer = layer[4]
                    with tf.variable_scope("mev_ts_attention"):
                        decoder_inputs = mev_ts_attention_layer(
                            decoder_inputs, kwargs['mev_ts_keys'], kwargs['mev_ts_attn_bias'], z=kwargs['mev_ts'])

                with tf.variable_scope("ffn"):
                    decoder_inputs = feed_forward_network(decoder_inputs, input_padding)

        return self.output_normalization(decoder_inputs)
