import tensorflow as tf
import tensorflow.contrib.rnn as tf_rnn
from tensorflow.contrib import seq2seq
from tensorflow.contrib.framework import nest
from tensorflow.contrib.seq2seq import FinalBeamSearchDecoderOutput

from models.common import sequence

OPS_NAME = 'decoder'


def prepare_decoder_inputs(target_words, start_token_id):
    batch_size = tf.shape(target_words)[0]
    start_tokens = tf.fill([batch_size, 1], start_token_id)
    inputs = tf.concat([start_tokens, target_words], axis=1)

    return inputs


def prepare_decoder_output(target_words, lengths, stop_token_id, pad_token_id):
    batch_size = tf.shape(target_words)[0]

    extra_pad = tf.fill([batch_size, 1], pad_token_id)
    target_words = tf.concat([target_words, extra_pad], axis=1)
    max_length = tf.shape(target_words)[1]

    update_indices = tf.range(0, batch_size) * max_length + (lengths)
    update_indices = tf.reshape(update_indices, [-1, 1])
    flatten = tf.reshape(target_words, [-1])

    updates = tf.fill([batch_size], stop_token_id)
    delta = tf.scatter_nd(update_indices, updates, tf.shape(flatten))

    outputs = flatten + delta
    outputs = tf.reshape(outputs, [-1, max_length])

    return outputs

class AttentionAugmentRNNCellFixed(tf_rnn.MultiRNNCell):
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

        output = tf.concat([x, attention], axis=1)
        new_states = (tuple(new_states) if self._state_is_tuple else tf.concat(new_states, 1))

        return output, new_states


def create_decoder_cell(agenda, base_sent_embeds, insert_word_embeds, delete_word_embeds,
                        base_length, iw_length, dw_length,
                        attn_dim, hidden_dim, num_layer,
                        enable_alignment_history=False, enable_dropout=False, dropout_keep=0.1,
                        no_insert_delete_attn=False):
    base_attn = seq2seq.BahdanauAttention(attn_dim, base_sent_embeds, base_length, name='src_attn')
    attns = [base_attn]
    if not no_insert_delete_attn:
        insert_attn = seq2seq.BahdanauAttention(attn_dim, insert_word_embeds, iw_length, name='insert_attn')
        delete_attn = seq2seq.BahdanauAttention(attn_dim, delete_word_embeds, dw_length, name='delete_attn')
        attns += [insert_attn, delete_attn]

    if no_insert_delete_attn:
        assert len(attns) == 1
    else:
        assert len(attns) == 3

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

    return decoder_cell


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


class DecoderOutputLayer(tf.layers.Layer):
    def __init__(self, embedding, beam_decoder=False, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embedding.shape[-1]
        self.vocab_size = embedding.shape[0]

        self.embedding = embedding
        self.vocab_projection_pos = tf.layers.Dense(self.embed_dim)
        self.vocab_projection_neg = tf.layers.Dense(self.embed_dim)

        self.beam_decoder = beam_decoder

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.vocab_size)

    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs)[0]
        beam_width = inputs.shape[1]

        if self.beam_decoder:
            inputs = tf.reshape(inputs, [batch_size * beam_width, inputs.shape[2]])

        vocab_query_pos = self.vocab_projection_pos(inputs)
        vocab_query_neg = self.vocab_projection_neg(inputs)

        vocab_logit_pos = tf.nn.relu(tf.matmul(vocab_query_pos, tf.transpose(self.embedding)))
        vocab_logit_neg = tf.nn.relu(tf.matmul(vocab_query_neg, tf.transpose(self.embedding)))

        logits = vocab_logit_pos - vocab_logit_neg

        if self.beam_decoder:
            logits = tf.reshape(logits, [batch_size, beam_width, logits.shape[1]])

        return logits


def train_decoder(agenda, embeddings,
                  dec_inputs, base_sent_hiddens, insert_word_embeds, delete_word_embeds,
                  dec_input_lengths, base_length, iw_length, dw_length,
                  attn_dim, hidden_dim, num_layer, swap_memory, enable_dropout=False, dropout_keep=1.,
                  no_insert_delete_attn=False):
    with tf.variable_scope(OPS_NAME, 'decoder', []):
        batch_size = tf.shape(base_sent_hiddens)[0]

        dec_inputs = tf.nn.embedding_lookup(embeddings, dec_inputs)
        helper = seq2seq.TrainingHelper(dec_inputs, dec_input_lengths, name='train_helper')

        cell = create_decoder_cell(
            agenda,
            base_sent_hiddens, insert_word_embeds, delete_word_embeds,
            base_length, iw_length, dw_length,
            attn_dim, hidden_dim, num_layer,
            enable_dropout=enable_dropout, dropout_keep=dropout_keep,
            no_insert_delete_attn=no_insert_delete_attn
        )

        output_layer = DecoderOutputLayer(embeddings)
        zero_states = create_trainable_zero_state(cell, batch_size)

        decoder = seq2seq.BasicDecoder(cell, helper, zero_states, output_layer)

        outputs, state, length = seq2seq.dynamic_decode(decoder, swap_memory=swap_memory)

        return outputs, state, length


def beam_eval_decoder(agenda, embeddings, start_token_id, stop_token_id,
                      base_sent_hiddens, insert_word_embeds, delete_word_embeds,
                      base_length, iw_length, dw_length,
                      attn_dim, hidden_dim, num_layer, maximum_iterations, beam_width, swap_memory,
                      enable_dropout=False, dropout_keep=1., no_insert_delete_attn=False):
    with tf.variable_scope(OPS_NAME, 'decoder', reuse=True):
        true_batch_size = tf.shape(base_sent_hiddens)[0]

        tiled_agenda = seq2seq.tile_batch(agenda, beam_width)

        tiled_base_sent = seq2seq.tile_batch(base_sent_hiddens, beam_width)
        tiled_insert_embeds = seq2seq.tile_batch(insert_word_embeds, beam_width)
        tiled_delete_embeds = seq2seq.tile_batch(delete_word_embeds, beam_width)

        tiled_src_lengths = seq2seq.tile_batch(base_length, beam_width)
        tiled_iw_lengths = seq2seq.tile_batch(iw_length, beam_width)
        tiled_dw_lengths = seq2seq.tile_batch(dw_length, beam_width)

        start_token_id = tf.cast(start_token_id, tf.int32)
        stop_token_id = tf.cast(stop_token_id, tf.int32)

        cell = create_decoder_cell(
            tiled_agenda,
            tiled_base_sent, tiled_insert_embeds, tiled_delete_embeds,
            tiled_src_lengths, tiled_iw_lengths, tiled_dw_lengths,
            attn_dim, hidden_dim, num_layer,
            enable_dropout=enable_dropout, dropout_keep=dropout_keep,
            no_insert_delete_attn=no_insert_delete_attn
        )

        output_layer = DecoderOutputLayer(embeddings, beam_decoder=True)
        zero_states = create_trainable_zero_state(cell, true_batch_size, beam_width)

        decoder = seq2seq.BeamSearchDecoder(
            cell,
            embeddings,
            tf.fill([true_batch_size], start_token_id),
            stop_token_id,
            zero_states,
            beam_width=beam_width,
            output_layer=output_layer,
            length_penalty_weight=0.0
        )

        return seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations, swap_memory=swap_memory)


def greedy_eval_decoder(agenda, embeddings, start_token_id, stop_token_id,
                        base_sent_hiddens, insert_word_embeds, delete_word_embeds,
                        base_length, iw_length, dw_length,
                        attn_dim, hidden_dim, num_layer, max_sentence_length, swap_memory,
                        enable_dropout=False, dropout_keep=1., no_insert_delete_attn=False):
    with tf.variable_scope(OPS_NAME, 'decoder', reuse=True):
        batch_size = tf.shape(base_sent_hiddens)[0]

        start_token_id = tf.cast(start_token_id, tf.int32)
        stop_token_id = tf.cast(stop_token_id, tf.int32)

        helper = seq2seq.GreedyEmbeddingHelper(embeddings,
                                               tf.fill([batch_size], start_token_id),
                                               stop_token_id)

        cell = create_decoder_cell(
            agenda,
            base_sent_hiddens, insert_word_embeds, delete_word_embeds,
            base_length, iw_length, dw_length,
            attn_dim, hidden_dim, num_layer,
            enable_dropout=enable_dropout, dropout_keep=dropout_keep,
            no_insert_delete_attn=no_insert_delete_attn
        )

        output_layer = DecoderOutputLayer(embeddings)
        zero_states = create_trainable_zero_state(cell, batch_size)

        decoder = seq2seq.BasicDecoder(
            cell,
            helper,
            zero_states,
            output_layer
        )

        outputs, state, lengths = seq2seq.dynamic_decode(decoder, maximum_iterations=max_sentence_length,
                                                         swap_memory=swap_memory)

        return outputs, state, lengths


def str_tokens(decoder_output, vocab_i2s):
    return vocab_i2s.lookup(
        tf.to_int64(sample_id(decoder_output))
    )


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


def last_hidden_state(decoder_output):
    return decoder_output[1]
