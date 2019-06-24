import tensorflow as tf

from models.common import graph_utils
from models.common.config import Config
from models.im_all_transformer.transformer import model_utils
from models.im_all_transformer.transformer.embedding_layer import EmbeddingSharedWeights
from models.im_all_transformer.transformer.transformer import EncoderStack, DecoderStack

OPS_NAME = 'edit_encoder'


class TransformerMicroEditExtractor(tf.layers.Layer):
    def __init__(self, embedding_layer, mev_projection, params, **kwargs):
        super().__init__(**kwargs)
        self.params = params
        is_training = graph_utils.is_training()

        encoder_config = Config.merge_to_new([params, params.encoder])
        decoder_config = Config.merge_to_new([params, params.decoder])

        self.target_encoder = EncoderStack(encoder_config.to_json(), is_training)
        self.mev_decoder = DecoderStack(decoder_config.to_json(), is_training)

        self.embedding_layer = embedding_layer
        self.mev_projection = mev_projection

        self.cls_tok_embedding = self.add_weight('cls_tok_embedding',
                                                 (self.params.hidden_size,),
                                                 dtype=tf.float32,
                                                 trainable=True)

        self.pooling_layer = tf.layers.Dense(self.params.hidden_size, activation='tanh', name='pooling_layer')

    def call(self, src, tgt=None, src_len=None, tgt_len=None, **kwargs):
        assert src is not None \
               and tgt is not None \
               and src_len is not None \
               and tgt_len is not None

        # First calculate transformer's input and paddings
        # [batch, length]
        tgt_padding = model_utils.get_padding_by_seq_len(tgt_len)
        # [batch, 1, 1, length]
        tgt_attention_bias = model_utils.get_padding_bias(None, tgt_padding)

        # [batch, length, hidden_size]
        embedded_tgt = self.embedding_layer(tgt)
        # [batch, length, hidden_size]
        embedded_tgt += model_utils.get_position_encoding(
            tf.shape(embedded_tgt)[1],
            self.params.hidden_size
        )

        # Add [CLS] token to the beginning of source sequence
        # [batch, length, hidden_size]
        embedded_src = self.embedding_layer(src)
        # [batch, length+1, hidden_size]
        extended_embedded_src = self._add_cls_token(embedded_src)
        extended_embedded_src += model_utils.get_position_encoding(
            tf.shape(extended_embedded_src)[1],
            self.params.hidden_size
        )
        extended_src_len = src_len + 1

        # [batch, length+1]
        extended_src_padding = model_utils.get_padding_by_seq_len(extended_src_len)
        # [batch, 1,1, length+1]
        extended_src_attention_bias = model_utils.get_padding_bias(None, extended_src_padding)

        # Encode Target
        # [batch, length, hidden_size]
        encoded_tgt = self._encode_tgt(embedded_tgt, tgt_padding, tgt_attention_bias)

        # Decode source using the encoded target
        # [batch, length+1, hidden_size]
        decoder_output = self._decode_micro_edit_vectors(extended_embedded_src, extended_src_padding,
                                                         extended_src_attention_bias,
                                                         encoded_tgt, tgt_attention_bias)

        if not graph_utils.is_training():
            tf.add_to_collection('TransformerMicroEditExtractor_Attentions', [
                self.target_encoder.self_attn_alignment_history,
                self.mev_decoder.self_attn_alignment_history,
                self.mev_decoder.enc_dec_attn_alignment_history,
            ])

        with tf.name_scope("pooler"):
            # We "pool" the model by simply taking the hidden state corresponding
            # to the first token.
            first_token_tensor = tf.squeeze(decoder_output[:, 0:1, :], axis=1)
            pooled = self.pooling_layer(first_token_tensor)

        # [batch, length, hidden_size]
        micro_ev = self.mev_projection(decoder_output[:, 1:, :])

        return encoded_tgt, tgt_attention_bias, pooled, micro_ev

    def _add_cls_token(self, embedded_seq):
        """
        Args:
            embedded_seq(Tensor): shape: [batch, length, hidden_size]

        Returns:
            Tensor with [batch, length+1, hidden_size]
        """
        # [1, hidden_size]
        cls_embed = tf.reshape(self.cls_tok_embedding, [1, -1])

        batch_size = tf.shape(embedded_seq)[0]

        # [batch, hidden_size]
        cls_embed = tf.tile(cls_embed, [batch_size, 1])

        # [batch, 1, hidden_size)
        cls_embed = tf.expand_dims(cls_embed, axis=1)

        return tf.concat([cls_embed, embedded_seq], axis=1)

    def _encode_tgt(self, embedded_tgt, tgt_padding, tgt_attention_bias):
        with tf.name_scope('encode_tgt'):
            if self.params.enable_dropout and self.params.layer_postprocess_dropout > 0.0:
                embedded_tgt = tf.nn.dropout(embedded_tgt, 1.0 - self.params.layer_postprocess_dropout)

            return self.target_encoder(embedded_tgt, tgt_attention_bias, tgt_padding)

    def _decode_micro_edit_vectors(self, embedded_src, src_padding, src_attention_bias,
                                   encoded_tgt, tgt_attention_bias):
        with tf.name_scope('decode_mev'):
            if self.params.enable_dropout and self.params.layer_postprocess_dropout > 0.0:
                embedded_src = tf.nn.dropout(embedded_src, 1 - self.params.layer_postprocess_dropout)

            outputs = self.mev_decoder(embedded_src, encoded_tgt,
                                       src_attention_bias, tgt_attention_bias,
                                       input_padding=src_padding)

            return outputs


class WordEmbeddingAccumulator(tf.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.projector = tf.layers.Dense(config.accumulated_dim,
                                         activation=None,
                                         use_bias=False,
                                         name='projector')

    # noinspection PyMethodOverriding
    def call(self, inp_embeds, inp_len, **kwargs):
        max_len = tf.shape(inp_embeds)[1]

        mask = tf.sequence_mask(inp_len, maxlen=max_len, dtype=tf.float32)
        mask = tf.expand_dims(mask, 2)

        acc = tf.reduce_sum(mask * inp_embeds, axis=1)
        acc = self.projector(acc)

        return acc


class EditEncoder(tf.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        self.embedding_layer = EmbeddingSharedWeights.get_from_graph()
        if config.editor.word_dim != config.editor.edit_encoder.extractor.hidden_size:
            self.embedding_layer = self.embedding_layer.get_projected(
                config.editor.edit_encoder.extractor.hidden_size)

        self.micro_ev_projection = tf.layers.Dense(
            config.editor.edit_encoder.micro_ev_dim,
            activation=config.editor.edit_encoder.get('mev_proj_activation_fn', None),
            use_bias=True,
            name='micro_ev_proj'
        )

        self.edit_vector_projection = tf.layers.Dense(
            config.editor.edit_encoder.edit_dim,
            activation=config.editor.edit_encoder.get('edit_vector_proj_activation_fn', None),
            use_bias=False,
            name='encoder_ev'
        )

        self.wa = WordEmbeddingAccumulator(config.editor.edit_encoder.word_acc)

        extractor_config = Config.merge_to_new([config.editor.transformer, config.editor.edit_encoder.extractor])
        self.mev_extractor = TransformerMicroEditExtractor(
            self.embedding_layer,
            self.micro_ev_projection,
            extractor_config
        )

    # noinspection PyMethodOverriding
    def call(self, src_word_ids, tgt_word_ids,
             insert_word_ids, common_word_ids,
             src_len, tgt_len, iw_len, cw_len, **kwargs):
        with tf.variable_scope('edit_encoder'):
            orig_embedding_layer = EmbeddingSharedWeights.get_from_graph()
            wa_inserted = self.wa(orig_embedding_layer(insert_word_ids), iw_len)
            wa_common = self.wa(orig_embedding_layer(common_word_ids), iw_len)

            if self.config.editor.enable_dropout and self.config.editor.dropout > 0.:
                wa_inserted = tf.nn.dropout(wa_inserted, 1. - self.config.editor.dropout)
                wa_common = tf.nn.dropout(wa_common, 1. - self.config.editor.dropout)

            outputs = self.mev_extractor(src_word_ids, tgt_word_ids, src_len, tgt_len)
            cnx_tgt, tgt_attn_bias, pooled_src, micro_evs_st = outputs

            outputs = self.mev_extractor(tgt_word_ids, src_word_ids, tgt_len, src_len)
            cnx_src, src_attn_bias, pooled_tgt, micro_evs_ts = outputs

            features = tf.concat([
                pooled_src,
                pooled_tgt,
                wa_inserted,
                wa_common
            ], axis=1)

            edit_vector = self.edit_vector_projection(features)

            if self.config.editor.enable_dropout and self.config.editor.dropout > 0.:
                edit_vector = tf.nn.dropout(edit_vector, 1. - self.config.editor.dropout)

            return edit_vector, (micro_evs_st, cnx_src, src_attn_bias), (micro_evs_ts, cnx_tgt, tgt_attn_bias)
