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

    def zero_state(self, batch_size, dtype, trainable=False):
        zero_states = super().zero_state(batch_size, dtype)
        if not trainable:
            return super().zero_state(batch_size, dtype)

        h, c = self._cells[0].state_size.cell_state
        name_prefix = 'zero_state'
        init_c = tf.get_variable(
            '%s_c' % (name_prefix),
            [1, c],
            initializer=tf.constant_initializer(0.0),
            trainable=True)

        init_h = tf.get_variable(
            '%s_h' % (name_prefix),
            [1, h],
            initializer=tf.constant_initializer(0.0),
            trainable=True
        )

        btm_layer_initial = tf_rnn.LSTMStateTuple(
            tf.tile(init_c, [batch_size, 1]),
            tf.tile(init_h, [batch_size, 1])
        )

        btm_layer_initial = seq2seq.AttentionWrapperState(
            btm_layer_initial,
            zero_states[0].attention,
            zero_states[0].time,
            zero_states[0].alignments,
            zero_states[0].alignment_history,
            zero_states[0].attention_state,
        )
        initial_variables = [btm_layer_initial]

        for i, (c, h) in enumerate(self.state_size[1:]):
            init_c = tf.get_variable(
                '%s_c_%s' % (name_prefix, i),
                [1, c],
                initializer=tf.constant_initializer(0.0),
                trainable=True)

            init_h = tf.get_variable(
                '%s_h_%s' % (name_prefix, i),
                [1, h],
                initializer=tf.constant_initializer(0.0),
                trainable=True
            )

            initial_variables.append(tf_rnn.LSTMStateTuple(
                tf.tile(init_c, [batch_size, 1]),
                tf.tile(init_h, [batch_size, 1])
            ))

        return tuple(initial_variables)

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


def create_decoder_cell(agenda, src_sent_embeds, insert_word_embeds, delete_word_embeds,
                        src_lengths, iw_length, dw_length,
                        attn_dim, hidden_dim, num_layer):
    src_attn = seq2seq.BahdanauAttention(attn_dim, src_sent_embeds, src_lengths, name='src_attn')
    insert_attn = seq2seq.BahdanauAttention(attn_dim, insert_word_embeds, iw_length, name='insert_attn')
    delete_attn = seq2seq.BahdanauAttention(attn_dim, delete_word_embeds, dw_length, name='delete_attn')

    bottom_cell = tf_rnn.LSTMCell(hidden_dim, name='bottom_cell')
    bottom_attn_cell = seq2seq.AttentionWrapper(
        bottom_cell,
        (src_attn, insert_attn, delete_attn),
        output_attention=False,
        name='bottom_attn_cell'
    )

    all_cells = [bottom_attn_cell]

    num_layer -= 1
    for i in range(num_layer):
        all_cells.append(tf_rnn.LSTMCell(hidden_dim, name='layer_%s' % (i + 1)))

    decoder_cell = AttentionAugmentRNNCell(all_cells)
    decoder_cell.set_agenda(agenda)

    return decoder_cell


class DecoderOutputLayer(tf.layers.Layer):
    def __init__(self, embedding, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embedding.shape[-1]
        self.vocab_size = embedding.shape[0]

        self.embedding = embedding
        self.vocab_projection_pos = tf.layers.Dense(self.embed_dim)
        self.vocab_projection_neg = tf.layers.Dense(self.embed_dim)

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


def train_decoder(agenda, embeddings, dec_inputs,
                  src_sent_embeds, insert_word_embeds, delete_word_embeds,
                  dec_input_lengths, src_lengths, iw_length, dw_length,
                  attn_dim, hidden_dim, num_layer):
    with tf.variable_scope(OPS_NAME, 'decoder', []):
        batch_size = tf.shape(src_sent_embeds)[0]

        helper = seq2seq.TrainingHelper(dec_inputs, dec_input_lengths, name='train_helper')

        cell = create_decoder_cell(
            agenda,
            src_sent_embeds, insert_word_embeds, delete_word_embeds,
            src_lengths, iw_length, dw_length,
            attn_dim, hidden_dim, num_layer
        )

        output_layer = DecoderOutputLayer(embeddings)
        zero_states = cell.zero_state(batch_size, tf.float32, trainable=True)
        decoder = seq2seq.BasicDecoder(cell, helper, zero_states, output_layer)

        outputs, _, _ = seq2seq.dynamic_decode(decoder)

        return outputs
