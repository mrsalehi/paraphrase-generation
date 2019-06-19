import tensorflow as tf
import tensorflow.contrib.rnn as tf_rnn
from tensorflow.contrib import seq2seq
from tensorflow.contrib.seq2seq import FinalBeamSearchDecoderOutput
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest

from models.common import sequence, vocab

OPS_NAME = 'decoder'


class AttentionAugmentRNNCell(tf_rnn.MultiRNNCell):
    def set_source_attn_index(self, index):
        self.source_attn_index = index

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

    def get_source_attention(self, state):
        attention = state[0].attention
        alignments = state[0].alignments[self.source_attn_index]
        return attention, alignments


class DecoderOutputLayer(tf.layers.Layer):
    def __init__(self, embedding, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embedding.shape[-1]
        self.vocab_size = embedding.shape[0]

        self.embedding = embedding
        self.vocab_projection_pos = tf.layers.Dense(self.embed_dim, name='vocab_projection_pos')
        self.vocab_projection_neg = tf.layers.Dense(self.embed_dim, name='vocab_projection_neg')

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.vocab_size)

    def call(self, inputs, **kwargs):
        vocab_query_pos = self.vocab_projection_pos(inputs)
        vocab_query_neg = self.vocab_projection_neg(inputs)

        vocab_logit_pos = tf.nn.relu(tf.matmul(vocab_query_pos, tf.transpose(self.embedding)))
        vocab_logit_neg = tf.nn.relu(tf.matmul(vocab_query_neg, tf.transpose(self.embedding)))

        logits = vocab_logit_pos - vocab_logit_neg

        return logits


class PointerGeneratorWrapper(tf.contrib.rnn.RNNCell):
    def __init__(self, cell, source_extend_tokens, max_oovs, output_layer, vocab_size,
                 source_attn_alignment_fn, **kwargs):
        super(PointerGeneratorWrapper, self).__init__(**kwargs)
        self.cell = cell
        self.source_extend_tokens = source_extend_tokens
        self.max_oovs = max_oovs
        self.output_layer = output_layer
        self.vocab_size = vocab_size
        self.source_attn_alignment_fn = source_attn_alignment_fn
        self._output_size = vocab_size + max_oovs
        self._pg_layer = Dense(1, tf.sigmoid, use_bias=True, name='pg_layer')

    def call(self, inputs, state):
        # Run the cell
        (attn_cell_outputs, cell_outputs), cell_state = self.cell(inputs, state)

        attention, alignments = self.source_attn_alignment_fn(cell_state)
        p_gen = self._pg_layer(tf.concat([attention, inputs, cell_state[-1].c, cell_outputs], axis=1))

        if self.output_layer is not None:
            cell_outputs = self.output_layer(cell_outputs)

        vocab_dist = p_gen * tf.nn.softmax(cell_outputs)
        alignments = (1 - p_gen) * alignments

        extended_vsize = self.vocab_size + self.max_oovs
        batch_size = tf.shape(vocab_dist)[0]
        extra_zeros = tf.zeros((batch_size, self.max_oovs))

        # batch * extend vocab size
        vocab_dists_extended = tf.concat(axis=-1, values=[vocab_dist, extra_zeros])

        batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
        input_seq_len = tf.shape(self.source_extend_tokens)[1]  # number of states we attend over
        batch_nums = tf.tile(batch_nums, [1, input_seq_len])  # shape (batch_size, attn_len)
        indices = tf.stack((tf.to_int64(batch_nums), self.source_extend_tokens), axis=2)  # shape (batch_size, enc_t, 2)

        attn_dists_projected = tf.scatter_nd(indices, alignments, [batch_size, extended_vsize])

        final_dists = attn_dists_projected + vocab_dists_extended

        return final_dists, cell_state

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
            It can be represented by an Integer, a TensorShape or a tuple of Integers
            or TensorShapes.
        """
        return self.cell.state_size

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self._output_size

    def zero_state(self, batch_size, dtype):
        return self.cell.zero_state(batch_size, dtype)


def create_decoder_cell(agenda, extended_base_words, oov, base_sent_hiddens, mev_st, mev_ts,
                        base_length, iw_length, dw_length,
                        vocab_size, attn_dim, hidden_dim, num_layer,
                        enable_alignment_history=False, enable_dropout=False, dropout_keep=1.,
                        no_insert_delete_attn=False, beam_width=None):
    base_attn = seq2seq.BahdanauAttention(attn_dim, base_sent_hiddens, base_length, name='base_attn')

    cnx_src, micro_evs_st = mev_st
    mev_st_attn = seq2seq.BahdanauAttention(attn_dim, cnx_src, iw_length, name='mev_st_attn')
    mev_st_attn._values = micro_evs_st

    attns = [base_attn, mev_st_attn]

    if not no_insert_delete_attn:
        cnx_tgt, micro_evs_ts = mev_ts
        mev_ts_attn = seq2seq.BahdanauAttention(attn_dim, cnx_tgt, dw_length, name='mev_ts_attn')
        mev_ts_attn._values = micro_evs_ts

        attns += [mev_ts_attn]

    is_training = tf.get_collection('is_training')[0]
    enable_alignment_history = not is_training

    bottom_cell = tf_rnn.LSTMCell(hidden_dim, name='bottom_cell')
    bottom_attn_cell = seq2seq.AttentionWrapper(
        bottom_cell,
        tuple(attns),
        alignment_history=enable_alignment_history,
        output_attention=False,
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
    decoder_cell.set_source_attn_index(0)

    output_layer = DecoderOutputLayer(vocab.get_embeddings())

    pg_cell = PointerGeneratorWrapper(
        decoder_cell,
        extended_base_words,
        50,
        output_layer,
        vocab_size,
        decoder_cell.get_source_attention,
        name='PointerGeneratorWrapper'
    )

    if beam_width:
        true_batch_size = tf.cast(tf.shape(base_sent_hiddens)[0] / beam_width, tf.int32)
    else:
        true_batch_size = tf.shape(base_sent_hiddens)[0]

    zero_state = create_trainable_zero_state(decoder_cell, true_batch_size, beam_width=beam_width)

    return pg_cell, zero_state


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
    def fn(orig_ids):
        orig_ids = tf.cast(orig_ids, tf.int64)
        in_vocab_ids = tf.where(
            tf.less(orig_ids, vocab_size),
            orig_ids,
            tf.ones_like(orig_ids) * vocab.OOV_TOKEN_ID
        )
        embeds = vocab.embed_tokens(in_vocab_ids)
        return embeds

    return fn


def create_embedding_fn_beam_decoder(vocab_size):
    def fn(orig_ids):
        orig_ids = tf.cast(orig_ids, tf.int64)

        in_vocab_ids = tf.where(
            tf.less(orig_ids, vocab_size),
            orig_ids,
            tf.ones_like(orig_ids) * vocab.OOV_TOKEN_ID
        )
        embeds = vocab.embed_tokens(in_vocab_ids)

        last_ids = tf.where(
            tf.equal(orig_ids, vocab.get_token_id(vocab.START_TOKEN)),
            tf.ones_like(orig_ids) * -1,
            orig_ids
        )
        last_ids = tf.cast(tf.expand_dims(last_ids, 2), tf.float32)

        cell_input = tf.concat([embeds, last_ids], axis=2)

        return cell_input

    return fn


def train_decoder(agenda, embeddings, extended_base_words, oov,
                  dec_inputs, dec_extended_inputs, base_sent_hiddens, insert_word_embeds, delete_word_embeds,
                  dec_input_lengths, base_length, iw_length, dw_length,
                  vocab_size, attn_dim, hidden_dim, num_layer, swap_memory, enable_dropout=False, dropout_keep=1.,
                  no_insert_delete_attn=False):
    with tf.variable_scope(OPS_NAME, 'decoder'):
        dec_input_embeds = vocab.embed_tokens(dec_inputs)
        helper = seq2seq.TrainingHelper(dec_input_embeds, dec_input_lengths, name='train_helper')

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


def beam_eval_decoder(agenda, embeddings, extended_base_words, oov,
                      start_token_id, stop_token_id,
                      base_sent_hiddens, insert_word_embeds, delete_word_embeds,
                      base_length, iw_length, dw_length,
                      vocab_size, attn_dim, hidden_dim, num_layer, max_sentence_length, beam_width, swap_memory,
                      enable_dropout=False, dropout_keep=1., no_insert_delete_attn=False):
    with tf.variable_scope(OPS_NAME, 'decoder', reuse=True):
        true_batch_size = tf.shape(base_sent_hiddens)[0]

        tiled_agenda = seq2seq.tile_batch(agenda, beam_width)
        tiled_extended_base_words = seq2seq.tile_batch(extended_base_words, beam_width)
        tiled_oov = seq2seq.tile_batch(oov, beam_width)

        tiled_base_sent = seq2seq.tile_batch(base_sent_hiddens, beam_width)
        tiled_insert_embeds = seq2seq.tile_batch(insert_word_embeds, beam_width)
        tiled_delete_embeds = seq2seq.tile_batch(delete_word_embeds, beam_width)

        tiled_src_lengths = seq2seq.tile_batch(base_length, beam_width)
        tiled_iw_lengths = seq2seq.tile_batch(iw_length, beam_width)
        tiled_dw_lengths = seq2seq.tile_batch(dw_length, beam_width)

        start_token_id = tf.cast(start_token_id, tf.int32)
        stop_token_id = tf.cast(stop_token_id, tf.int32)

        cell, zero_states = create_decoder_cell(
            tiled_agenda, tiled_extended_base_words, tiled_oov,
            tiled_base_sent, tiled_insert_embeds, tiled_delete_embeds,
            tiled_src_lengths, tiled_iw_lengths, tiled_dw_lengths,
            vocab_size, attn_dim, hidden_dim, num_layer,
            enable_dropout=enable_dropout, dropout_keep=dropout_keep,
            no_insert_delete_attn=no_insert_delete_attn, beam_width=beam_width
        )

        decoder = seq2seq.BeamSearchDecoder(
            cell,
            create_embedding_fn(vocab_size),
            tf.fill([true_batch_size], start_token_id),
            stop_token_id,
            zero_states,
            beam_width=beam_width,
            length_penalty_weight=0.0
        )

        return seq2seq.dynamic_decode(decoder, maximum_iterations=max_sentence_length, swap_memory=swap_memory)


def str_tokens(decoder_output, vocab_i2s, vocab_size, oov):
    sample_ids = sample_id(decoder_output)

    batch_size = tf.shape(sample_ids)[0]
    max_sent_len = tf.shape(sample_ids)[1]

    batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
    batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
    batch_nums = tf.tile(batch_nums, [1, max_sent_len])  # shape (batch_size, max_sent_len)

    oov_ids = tf.maximum(0, sample_ids - vocab_size)
    if isinstance(decoder_output[0], FinalBeamSearchDecoderOutput):
        beam_width = tf.shape(oov_ids)[-1]
        batch_nums = tf.tile(tf.expand_dims(batch_nums, 2), [1, 1, beam_width])
        oov_ids = tf.stack([batch_nums, oov_ids], axis=3)
    else:
        oov_ids = tf.stack([batch_nums, oov_ids], axis=2)  # shape (batch_size, max_sent_len, 2)

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


def alignment_history(decoder_output):
    final_state = decoder_output[1]
    history = final_state[0].alignment_history
    return history


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
