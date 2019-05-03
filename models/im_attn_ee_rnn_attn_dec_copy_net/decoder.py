import collections

import tensorflow as tf
import tensorflow.contrib.rnn as tf_rnn
from tensorflow.contrib import seq2seq
from tensorflow.contrib.framework.python.framework import tensor_util
from tensorflow.contrib.seq2seq import FinalBeamSearchDecoderOutput
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest

from models.common import sequence, vocab
from models.neural_editor.decoder import DecoderOutputLayer

OPS_NAME = 'decoder'


class CopyNetWrapperState(
    collections.namedtuple("CopyNetWrapperState", ("cell_state", "prob_c"))):
    def clone(self, **kwargs):
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, tf.Tensor) and isinstance(new, tf.Tensor):
                return tensor_util.with_same_shape(old, new)
            return new

        return nest.map_structure(
            with_same_shape,
            self,
            super(CopyNetWrapperState, self)._replace(**kwargs))


class CopyNetWrapper(tf.contrib.rnn.RNNCell):
    '''
    A copynet RNN cell wrapper
    '''

    def __init__(self, cell, source_extend_tokens, max_oovs, encode_output, output_layer, vocab_size, name=None):
        '''
        Args:
            - cell: the decoder cell
            - source_extend_tokens: input tokens with oov word ids
            - max_oovs: max number of oov words in each batch
            - encode_output: the output of encoder cell
            - output_layer: the layer used to map decoder output to vocab distribution
            - vocab_size: this is target vocab size
        '''
        super(CopyNetWrapper, self).__init__(name=name)
        self.cell = cell
        self.source_extend_tokens = source_extend_tokens
        self.encode_output = encode_output
        self.max_oovs = max_oovs
        self.output_layer = output_layer
        self._output_size = vocab_size + max_oovs
        self.vocab_size = vocab_size
        self.copy_layer = Dense(self.cell.output_size[1], activation=tf.tanh, use_bias=False, name="Copy_Weight")

    def _get_input_embeddings(self, ids):
        ids = tf.where(
            tf.less(ids, self.vocab_size),
            ids, tf.ones_like(ids) * vocab.OOV_TOKEN_ID
        )

        return vocab.embed_tokens(ids)

    def call(self, inputs, state):
        prob_c = state.prob_c
        cell_state = state.cell_state

        last_ids = tf.cast(inputs[:, 300], tf.int64)
        inputs = inputs[:, :300]

        # get selective read
        # batch * input length
        mask = tf.cast(tf.equal(tf.expand_dims(last_ids, 1), self.source_extend_tokens), tf.float32)
        mask_sum = tf.reduce_sum(mask, axis=1)
        mask = tf.where(tf.less(mask_sum, 1e-7), mask, mask / tf.expand_dims(mask_sum, axis=1))
        pt = mask * prob_c
        selective_read = tf.einsum("ijk,ij->ik", self.encode_output, pt)

        inputs = tf.concat([inputs, selective_read], axis=-1)
        (attn_outputs, outputs), cell_state = self.cell(inputs, cell_state)

        # this is generate mode
        vocab_dist = self.output_layer(attn_outputs)

        # this is copy mode
        # batch * length * output size
        copy_score = self.copy_layer(self.encode_output)

        # batch * length
        copy_score = tf.einsum("ijk,ik->ij", copy_score, outputs)

        extended_vsize = self.vocab_size + self.max_oovs

        batch_size = tf.shape(vocab_dist)[0]
        extra_zeros = tf.zeros((batch_size, self.max_oovs))
        # batch * extend vocab size
        vocab_dists_extended = tf.concat(axis=-1, values=[vocab_dist, extra_zeros])

        # this part is same as that of point generator, but using einsum is much simpler.
        source_mask = tf.one_hot(self.source_extend_tokens, extended_vsize)
        attn_dists_projected = tf.einsum("ijn,ij->in", source_mask, copy_score)

        final_dist = vocab_dists_extended + attn_dists_projected

        # this is used to calculate p(y_t,c|.)
        # safe softmax
        final_dist_max = tf.expand_dims(tf.reduce_max(final_dist, axis=1), axis=1)
        final_dist_exp = tf.reduce_sum(tf.exp(final_dist - final_dist_max), axis=1)
        p_c = tf.exp(attn_dists_projected - final_dist_max) / tf.expand_dims(final_dist_exp, axis=1)
        p_c = tf.einsum("ijn,in->ij", source_mask, p_c)

        state = CopyNetWrapperState(cell_state=cell_state, prob_c=p_c)
        return final_dist, state

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
            It can be represented by an Integer, a TensorShape or a tuple of Integers
            or TensorShapes.
        """
        return CopyNetWrapperState(cell_state=self.cell.state_size,
                                   prob_c=self.source_extend_tokens.shape[-1].value)

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self._output_size

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            cell_state = self.cell.zero_state(batch_size, dtype)
            prob_c = tf.zeros([batch_size, tf.shape(self.encode_output)[1]], tf.float32)
            return CopyNetWrapperState(cell_state=cell_state, prob_c=prob_c)


class AttentionAugmentRNNCell(tf_rnn.MultiRNNCell):
    def set_agenda(self, agenda):
        self.agenda = agenda

    def call(self, inputs, state):
        cur_state_pos = 0
        x = inputs
        new_states = []

        attn_cell = self._cells[0]
        with tf.variable_scope("cell_%d" % 0):
            if self._state_is_tuple:
                if not nest.is_sequence(state):
                    raise ValueError(
                        "Expected state to be a tuple of length %d, but received: %s" %
                        (len(self.state_size), state))
                cur_state = state[0]
            else:
                cur_state = tf.slice(state, [0, cur_state_pos], [-1, attn_cell.state_size])
                cur_state_pos += attn_cell.state_size

            if not isinstance(cur_state, seq2seq.AttentionWrapperState):
                raise ValueError("First state should be instance of AttentionWrapperState")

            attention = cur_state.attention
            rnn_input = tf.concat([x, self.agenda], axis=-1)
            h, new_state = attn_cell(rnn_input, cur_state)
            new_states.append(new_state)
            new_attention = new_state.attention

            # no skip connection on
            x = h

        for i, cell in enumerate(self._cells[1:]):
            i = i + 1
            with tf.variable_scope("cell_%d" % (i)):
                if self._state_is_tuple:
                    if not nest.is_sequence(state):
                        raise ValueError(
                            "Expected state to be a tuple of length %d, but received: %s" %
                            (len(self.state_size), state))
                    cur_state = state[i]
                else:
                    cur_state = tf.slice(state, [0, cur_state_pos], [-1, cell.state_size])
                    cur_state_pos += cell.state_size

                rnn_input = tf.concat([x, self.agenda, attention], axis=-1)
                h, new_state = cell(rnn_input, cur_state)
                new_states.append(new_state)

                # add residual connection from cell input to cell output
                x = x + h

        output = tf.concat([x, new_attention], axis=1)
        output = (output, x)
        new_states = (tuple(new_states) if self._state_is_tuple else tf.concat(new_states, 1))

        return output, new_states

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return (
            self._cells[-1].output_size + self._cells[0].state_size.attention,
            self._cells[-1].output_size
        )


def create_decoder_cell(agenda, extended_base_words, oov, base_sent_hiddens, mev_st, mev_ts,
                        base_length, iw_length, dw_length,
                        vocab_size, attn_dim, hidden_dim, num_layer,
                        enable_alignment_history=False, enable_dropout=False, dropout_keep=1.,
                        no_insert_delete_attn=False, beam_width=None):
    batch_size = tf.shape(base_sent_hiddens)[0]

    base_attn = seq2seq.BahdanauAttention(attn_dim, base_sent_hiddens, base_length, name='src_attn')

    cnx_src, micro_evs_st = mev_st
    mev_st_attn = seq2seq.BahdanauAttention(attn_dim, cnx_src, iw_length, name='mev_st_attn')
    mev_st_attn._values = micro_evs_st

    attns = [base_attn, mev_st_attn]

    if not no_insert_delete_attn:
        cnx_tgt, micro_evs_ts = mev_ts
        mev_ts_attn = seq2seq.BahdanauAttention(attn_dim, cnx_tgt, dw_length, name='mev_ts_attn')
        mev_ts_attn._values = micro_evs_ts

        attns += [mev_ts_attn]

    bottom_cell = tf_rnn.LSTMCell(hidden_dim, name='bottom_cell')
    bottom_attn_cell = seq2seq.AttentionWrapper(
        bottom_cell,
        tuple(attns),
        output_attention=False,
        alignment_history=enable_alignment_history,
        name='att_bottom_cell'
    )

    all_cells = [bottom_attn_cell]

    num_layer -= 1
    for i in range(num_layer):
        cell = tf_rnn.LSTMCell(hidden_dim, name='layer_%s' % (i + 1))
        if enable_dropout and dropout_keep < 1.:
            cell = tf_rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep)

        all_cells.append(cell)

    decoder_cell = AttentionAugmentRNNCell(all_cells)
    decoder_cell.set_agenda(agenda)

    zero_state = create_trainable_zero_state(decoder_cell, batch_size, beam_width=beam_width)

    beam_decoder = beam_width is not None
    output_layer = DecoderOutputLayer(vocab.get_embeddings(), beam_decoder=beam_decoder)

    # max_oov_length = tf.shape(oov)[-1]
    copy_net_cell = CopyNetWrapper(
        decoder_cell,
        extended_base_words,
        50,
        base_sent_hiddens,
        output_layer,
        vocab_size,
        name='CopyNetWrapper'
    )

    zero_state = copy_net_cell.zero_state(batch_size, tf.float32).clone(cell_state=zero_state)

    return copy_net_cell, zero_state


def create_trainable_zero_state(decoder_cell, batch_size, beam_width=None):
    if beam_width:
        default_initial_states = decoder_cell.zero_state(batch_size * beam_width, tf.float32)
    else:
        default_initial_states = decoder_cell.zero_state(batch_size, tf.float32)
    state_sizes = decoder_cell.state_size

    name_prefix = 'zero_state'

    attn_cell_state_size = state_sizes[0].cell_state
    attn_trainable_cs = sequence.create_trainable_lstm_initial_state(
        attn_cell_state_size,
        batch_size,
        'zero_state_btm_',
        beam_width
    )
    attn_init_state = default_initial_states[0].clone(cell_state=attn_trainable_cs)

    init_states = [attn_init_state] + list(sequence.create_trainable_initial_states_ss(
        batch_size,
        state_sizes[1:],
        name_prefix,
        beam_width
    ))

    return tuple(init_states)


def create_embedding_fn(vocab_size):
    def fn(ids):
        ids = tf.cast(ids, tf.int64)
        ids = tf.where(
            tf.less(ids, vocab_size),
            ids, tf.ones_like(ids) * vocab.get_token_id(vocab.UNKNOWN_TOKEN)
        )
        embeds = vocab.embed_tokens(ids)
        inputs = tf.concat([embeds, tf.cast(tf.expand_dims(ids, 1), tf.float32)], axis=1)
        return inputs

    return fn


def train_decoder(agenda, embeddings, extended_base_words, oov,
                  dec_inputs, base_sent_hiddens, insert_word_embeds, delete_word_embeds,
                  dec_input_lengths, base_length, iw_length, dw_length,
                  vocab_size, attn_dim, hidden_dim, num_layer, swap_memory, enable_dropout=False, dropout_keep=1.,
                  no_insert_delete_attn=False):
    with tf.variable_scope(OPS_NAME, 'decoder', []):
        dec_inputs = tf.where(
            tf.less(dec_inputs, vocab_size),
            dec_inputs, tf.ones_like(dec_inputs) * vocab.OOV_TOKEN_ID
        )
        dec_input_embeds = vocab.embed_tokens(dec_inputs)

        inputs = tf.concat([dec_input_embeds, tf.cast(tf.expand_dims(dec_inputs, 2), tf.float32)], axis=2)
        helper = seq2seq.TrainingHelper(inputs, dec_input_lengths, name='train_helper')

        cell, zero_states = create_decoder_cell(
            agenda, extended_base_words, oov,
            base_sent_hiddens, insert_word_embeds, delete_word_embeds,
            base_length, iw_length, dw_length,
            vocab_size, attn_dim, hidden_dim, num_layer,
            enable_dropout=enable_dropout, dropout_keep=dropout_keep,
            no_insert_delete_attn=no_insert_delete_attn
        )

        decoder = seq2seq.BasicDecoder(cell, helper, zero_states)
        outputs, state, length = seq2seq.dynamic_decode(decoder, swap_memory=swap_memory)

        return outputs, state, length


def greedy_eval_decoder(agenda, embeddings, extended_base_words, oov,
                        start_token_id, stop_token_id,
                        base_sent_hiddens, insert_word_embeds, delete_word_embeds,
                        base_length, iw_length, dw_length,
                        vocab_size, attn_dim, hidden_dim, num_layer, max_sentence_length, swap_memory,
                        enable_dropout=False, dropout_keep=1., no_insert_delete_attn=False):
    with tf.variable_scope(OPS_NAME, 'decoder', reuse=True):
        batch_size = tf.shape(base_sent_hiddens)[0]

        start_token_id = tf.cast(start_token_id, tf.int32)
        stop_token_id = tf.cast(stop_token_id, tf.int32)

        helper = seq2seq.GreedyEmbeddingHelper(create_embedding_fn(vocab_size),
                                               tf.fill([batch_size], start_token_id),
                                               stop_token_id)

        cell, zero_states = create_decoder_cell(
            agenda, extended_base_words, oov,
            base_sent_hiddens, insert_word_embeds, delete_word_embeds,
            base_length, iw_length, dw_length,
            vocab_size, attn_dim, hidden_dim, num_layer,
            enable_dropout=enable_dropout, dropout_keep=dropout_keep,
            no_insert_delete_attn=no_insert_delete_attn
        )

        decoder = seq2seq.BasicDecoder(
            cell,
            helper,
            zero_states
        )

        outputs, state, lengths = seq2seq.dynamic_decode(decoder, maximum_iterations=max_sentence_length,
                                                         swap_memory=swap_memory)

        return outputs, state, lengths


# def beam_eval_decoder(agenda, embeddings, start_token_id, stop_token_id,
#                       base_sent_hiddens, insert_word_embeds, delete_word_embeds,
#                       base_length, iw_length, dw_length,
#                       attn_dim, hidden_dim, num_layer, maximum_iterations, beam_width, swap_memory,
#                       enable_dropout=False, dropout_keep=1., no_insert_delete_attn=False):
#     with tf.variable_scope(OPS_NAME, 'decoder', reuse=True):
#         true_batch_size = tf.shape(base_sent_hiddens)[0]
#
#         tiled_agenda = seq2seq.tile_batch(agenda, beam_width)
#
#         tiled_base_sent = seq2seq.tile_batch(base_sent_hiddens, beam_width)
#         tiled_insert_embeds = seq2seq.tile_batch(insert_word_embeds, beam_width)
#         tiled_delete_embeds = seq2seq.tile_batch(delete_word_embeds, beam_width)
#
#         tiled_src_lengths = seq2seq.tile_batch(base_length, beam_width)
#         tiled_iw_lengths = seq2seq.tile_batch(iw_length, beam_width)
#         tiled_dw_lengths = seq2seq.tile_batch(dw_length, beam_width)
#
#         start_token_id = tf.cast(start_token_id, tf.int32)
#         stop_token_id = tf.cast(stop_token_id, tf.int32)
#
#         cell = create_decoder_cell(
#             tiled_agenda,
#             tiled_base_sent, tiled_insert_embeds, tiled_delete_embeds,
#             tiled_src_lengths, tiled_iw_lengths, tiled_dw_lengths,
#             attn_dim, hidden_dim, num_layer,
#             enable_dropout=enable_dropout, dropout_keep=dropout_keep,
#             no_insert_delete_attn=no_insert_delete_attn
#         )
#
#         output_layer = DecoderOutputLayer(embeddings, beam_decoder=True)
#         zero_states = create_trainable_zero_state(cell, true_batch_size, beam_width)
#
#         decoder = seq2seq.BeamSearchDecoder(
#             cell,
#             embeddings,
#             tf.fill([true_batch_size], start_token_id),
#             stop_token_id,
#             zero_states,
#             beam_width=beam_width,
#             output_layer=output_layer,
#             length_penalty_weight=0.0
#         )
#
#         return seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations, swap_memory=swap_memory)


def str_tokens(decoder_output, vocab_i2s, vocab_size, oov):
    sample_ids = sample_id(decoder_output)

    batch_size = tf.shape(sample_ids)[0]
    max_sent_len = tf.shape(sample_ids)[1]

    batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
    batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
    batch_nums = tf.tile(batch_nums, [1, max_sent_len])  # shape (batch_size, max_sent_len)

    oov_ids = tf.maximum(0, sample_ids - vocab_size)
    oov_ids = tf.stack((batch_nums, oov_ids), axis=2)  # shape (batch_size, max_sent_len, 2)

    oov_tokens = tf.gather_nd(oov, oov_ids)
    lookup = vocab_i2s.lookup(tf.to_int64(sample_id(decoder_output)))

    tokens = tf.where(tf.less(sample_ids, vocab_size), lookup, oov_tokens)

    return tokens


def attention_score(decoder_output):
    final_attention_state = decoder_output[1][0]
    alignments = final_attention_state.alignment_history

    def convert(t):
        return tf.transpose(t.stack(), [1, 0, 2])

    if isinstance(alignments, tuple):
        return tuple([convert(t) for t in alignments])

    return convert(alignments)


def rnn_output(decoder_output):
    output = decoder_output[0].rnn_output
    return output


def sample_id(decoder_output):
    if isinstance(decoder_output[0], FinalBeamSearchDecoderOutput):
        output = decoder_output[0].predicted_ids
    else:
        output = decoder_output[0].sample_id
    return output


def seq_length(decoder_output):
    if isinstance(decoder_output[0], FinalBeamSearchDecoderOutput):
        return decoder_output[1].lengths

    return decoder_output[2]
