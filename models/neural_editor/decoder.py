import tensorflow as tf
from tensorflow.contrib.framework import nest
import tensorflow.contrib.rnn as tf_rnn
from tensorflow.contrib import seq2seq

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
            with tf.variable_scope("cell_%d" % i):
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

        new_states = (tuple(new_states) if self._state_is_tuple else tf.concat(new_states, 1))

        return h, new_states


def create_decoder_cell(agenda, src_sent_embeds, insert_word_embeds, delete_word_embeds,
                        src_lengths, iw_length, dw_length,
                        attn_dim, hidden_dim, num_layer):
    src_attn = seq2seq.BahdanauAttention(attn_dim, src_sent_embeds, src_lengths, name='src_attn')
    insert_attn = seq2seq.BahdanauAttention(attn_dim, insert_word_embeds, iw_length, name='insert_attn')
    delete_attn = seq2seq.BahdanauAttention(attn_dim, delete_word_embeds, dw_length, name='delete_attn')

    bottom_cell = tf_rnn.LSTMCell(hidden_dim, name='bottom_cell')
    bottom_attn_cell = seq2seq.AttentionWrapper(
        bottom_cell,
        [src_attn, insert_attn, delete_attn],
        name='bottom_attn_cell'
    )

    all_cells = [bottom_attn_cell]

    num_layer -= 1
    for i in range(num_layer):
        all_cells.append(tf_rnn.LSTMCell(hidden_dim, name='layer_%s' % (i + 1)))

    decoder_cell = AttentionAugmentRNNCell(all_cells)
    decoder_cell.set_agenda(agenda)

    return decoder_cell


def train_decoder():
    with tf.variable_scope(OPS_NAME, 'decoder', []):
        pass
