import tensorflow as tf

from models.common import graph_utils
from models.common.config import Config
from models.im_all_transformer.transformer import model_utils
from models.im_all_transformer.transformer.embedding_layer import EmbeddingSharedWeights
from models.im_all_transformer.transformer.transformer import EncoderStack, DecoderStack
from models.neural_editor.edit_encoder import sample_vMF

OPS_NAME = 'edit_encoder'


class TransformerMicroEditExtractor(tf.layers.Layer):
    def __init__(self, embedding_layer, mev_projection, params, **kwargs):
        super().__init__(**kwargs)
        self.params = params
        is_training = graph_utils.is_training()

        encoder_config = Config.merge([params, params.encoder])
        decoder_config = Config.merge([params, params.decoder])

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

        initializer = tf.variance_scaling_initializer(
            self.params.initializer_gain, mode="fan_avg", distribution="uniform")

        with tf.variable_scope("TMEV", initializer=initializer):
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

            with tf.name_scope("pooler"):
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                # [batch, hidden_size]
                first_token_tensor = tf.squeeze(decoder_output[:, 0:1, :], axis=1)
                pooled = self.pooling_layer(first_token_tensor)

            # [batch, length, hidden_size]
            micro_ev = self.mev_projection(decoder_output[:, 1:, :])

            return encoded_tgt, pooled, micro_ev

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


class ProjectedEmbedding(tf.layers.Layer):
    def __init__(self, hidden_dim, embedding_layer, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer: EmbeddingSharedWeights = embedding_layer
        self.embedding_proj = tf.layers.Dense(hidden_dim,
                                              activation=None,
                                              use_bias=False,
                                              name='embedding_proj')

    def call(self, inputs, **kwargs):
        embeddings = self.embedding_layer(inputs)
        projected = self.embedding_proj(embeddings)

        return projected


def wa_accumulator(insert_words, delete_words, iw_lengths, dw_lengths, edit_dim):
    max_len = tf.shape(insert_words)[1]
    mask = tf.sequence_mask(iw_lengths, maxlen=max_len, dtype=tf.float32)
    mask = tf.expand_dims(mask, 2)
    insert_embed = tf.reduce_sum(mask * insert_words, axis=1)

    max_len = tf.shape(delete_words)[1]
    mask = tf.sequence_mask(dw_lengths, maxlen=max_len, dtype=tf.float32)
    mask = tf.expand_dims(mask, 2)
    delete_embed = tf.reduce_sum(mask * delete_words, axis=1)

    linear_prenoise = tf.make_template('linear_prenoise', tf.layers.dense,
                                       units=edit_dim,
                                       activation=None,
                                       use_bias=False)
    insert_embed = linear_prenoise(insert_embed)
    delete_embed = linear_prenoise(delete_embed)

    return insert_embed, delete_embed


def attn_encoder(src_word_ids, tgt_word_ids, insert_embeds, common_embeds,
                 src_len, tgt_len, iw_len, cw_len,
                 config):
    with tf.variable_scope(OPS_NAME):
        wa_inserted, wa_common = wa_accumulator(
            insert_embeds, common_embeds,
            iw_len, cw_len,
            config.editor.edit_encoder.wa_dim
        )

        if config.editor.enable_dropout and config.editor.dropout_keep < 1.:
            wa_inserted = tf.nn.dropout(wa_inserted, config.editor.dropout_keep)
            wa_common = tf.nn.dropout(wa_common, config.editor.dropout_keep)

        if config.editor.word_dim != config.editor.edit_encoder.transformer.hidden_size:
            embedding_layer = ProjectedEmbedding(
                config.editor.edit_encoder.transformer.hidden_size,
                EmbeddingSharedWeights.get_from_graph()
            )
        else:
            embedding_layer = EmbeddingSharedWeights.get_from_graph()

        micro_ev_projection = tf.layers.Dense(
            config.editor.edit_encoder.micro_ev_dim,
            activation=config.editor.edit_encoder.get('mev_proj_activation_fn', None),
            use_bias=True,
            name='micro_ev_proj'
        )
        params = Config.merge([config.editor.transformer, config.editor.edit_encoder.transformer])

        mev_extractor = TransformerMicroEditExtractor(embedding_layer, micro_ev_projection, params)

        cnx_tgt, pooled_src, micro_evs_st = mev_extractor(src_word_ids, tgt_word_ids, src_len, tgt_len)
        cnx_src, pooled_tgt, micro_evs_ts = mev_extractor(tgt_word_ids, src_word_ids, tgt_len, src_len)

        features = tf.concat([
            pooled_src,
            pooled_tgt,
            wa_inserted,
            wa_common
        ], axis=1)

        edit_vector = tf.layers.dense(features, config.editor.edit_encoder.edit_dim, use_bias=False, name='encoder_ev')

        if config.editor.enable_dropout and config.editor.dropout_keep < 1.:
            edit_vector = tf.nn.dropout(edit_vector, config.editor.dropout_keep)

        return edit_vector, (cnx_src, micro_evs_st), (cnx_tgt, micro_evs_ts)
