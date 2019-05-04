import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _compute_attention
from tensorflow.python import ops
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.python.util import nest


class PGAttentionWrapper(seq2seq.AttentionWrapper):
    def __init__(self, cell,
                 attention_mechanism,
                 attention_layer_size=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=True,
                 initial_cell_state=None,
                 name=None,
                 coverage=False):
        super(PGAttentionWrapper, self).__init__(
            cell,
            attention_mechanism,
            attention_layer_size,
            alignment_history,
            cell_input_fn,
            output_attention,
            initial_cell_state,
            name)
        self.coverage = coverage

    def zero_state(self, batch_size, dtype):
        """Return an initial (zero) state tuple for this `AttentionWrapper`.
        **NOTE** Please see the initializer documentation for details of how
        to call `zero_state` if using an `AttentionWrapper` with a
        `BeamSearchDecoder`.
        Args:
        batch_size: `0D` integer tensor: the batch size.
        dtype: The internal state data type.
        Returns:
        An `AttentionWrapperState` tuple containing zeroed out tensors and,
        possibly, empty `TensorArray` objects.
        Raises:
        ValueError: (or, possibly at runtime, InvalidArgument), if
            `batch_size` does not match the output size of the encoder passed
            to the wrapper object at initialization time.
        """
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            error_message = (
                    "When calling zero_state of AttentionWrapper %s: " % self._base_name +
                    "Non-matching batch sizes between the memory "
                    "(encoder output) and the requested batch size.  Are you using "
                    "the BeamSearchDecoder?  If so, make sure your encoder output has "
                    "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
                    "the batch_size= argument passed to zero_state is "
                    "batch_size * beam_width.")
            with tf.control_dependencies(
                    self._batch_size_checks(batch_size, error_message)):
                cell_state = nest.map_structure(
                    lambda s: tf.identity(s, name="checked_cell_state"),
                    cell_state)
            return tf.contrib.seq2seq.AttentionWrapperState(
                cell_state=cell_state,
                time=tf.zeros([], dtype=tf.int32),
                attention=_zero_state_tensors(self._attention_layer_size, batch_size,
                                              dtype),
                alignments=self._item_or_tuple(
                    attention_mechanism.initial_alignments(batch_size, dtype)
                    for attention_mechanism in self._attention_mechanisms),
                # since we need to read the alignment history several times, so we need set clear_after_read to False
                alignment_history=self._item_or_tuple(
                    tf.TensorArray(dtype=dtype, size=0, clear_after_read=False, dynamic_size=True)
                    if self._alignment_history else ()
                    for _ in self._attention_mechanisms),
                attention_state=self._item_or_tuple(
                    attention_mechanism.initial_state(batch_size, dtype)
                    for attention_mechanism in self._attention_mechanisms)
            )

    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.
        - Step 1: Mix the `inputs` and previous step's `attention` output via
            `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
            `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
            alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output
            and context through the attention layer (a linear layer with
            `attention_layer_size` outputs).
        Args:
            inputs: (Possibly nested tuple of) Tensor, the input at this time step.
            state: An instance of `AttentionWrapperState` containing
            tensors from the previous time step.
        Returns:
            A tuple `(attention_or_cell_output, next_state)`, where:
            - `attention_or_cell_output` depending on `output_attention`.
            - `next_state` is an instance of `AttentionWrapperState`
                containing the state calculated at this time step.
        Raises:
            TypeError: If `state` is not an instance of `AttentionWrapperState`.
        """
        if not isinstance(state, tf.contrib.seq2seq.AttentionWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                            "Received type %s instead." % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (
                cell_output.shape[0].value or tf.shape(cell_output)[0])
        error_message = (
                "When applying AttentionWrapper %s: " % self.name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the query (decoder output).  Are you using "
                "the BeamSearchDecoder?  You may need to tile your memory input via "
                "the tf.contrib.seq2seq.tile_batch function with argument "
                "multiple=beam_width.")
        with tf.control_dependencies(
                self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = tf.identity(
                cell_output, name="checked_cell_output")

        if self._is_multi:
            previous_alignments = state.alignments
            previous_alignment_history = state.alignment_history
        else:
            previous_alignments = [state.alignments]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_histories = []

        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            if self.coverage:
                # if we use coverage mode, previous alignments is coverage vector
                # alignment history stack has shape:  decoder time * batch * atten_len
                # convert it to coverage vector
                previous_alignments[i] = tf.cond(
                    previous_alignment_history[i].size() > 0,
                    lambda: tf.reduce_sum(tf.transpose(previous_alignment_history[i].stack(), [1, 2, 0]), axis=2),
                    lambda: tf.zeros_like(previous_alignments[i]))
            # debug
            # previous_alignments[i] = tf.Print(previous_alignments[i],[previous_alignment_history[i].size(), tf.shape(previous_alignments[i]),previous_alignments[i]],message="atten wrapper:")
            attention, alignments, _ = _compute_attention(
                attention_mechanism, cell_output, previous_alignments[i],
                self._attention_layers[i] if self._attention_layers else None)
            alignment_history = previous_alignment_history[i].write(
                state.time, alignments) if self._alignment_history else ()

            all_alignments.append(alignments)
            all_histories.append(alignment_history)
            all_attentions.append(attention)

        attention = tf.concat(all_attentions, 1)
        next_state = seq2seq.AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(all_histories),
            attention_state=state.attention_state
        )

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state


def _pg_bahdanau_score(processed_query, keys, coverage, coverage_vector):
    """Implements Bahdanau-style (additive) scoring function.
    Args:
        processed_query: Tensor, shape `[batch_size, num_units]` to compare to keys.
        keys: Processed memory, shape `[batch_size, max_time, num_units]`.
        coverage: Whether to use coverage mode.
        coverage_vector: only used when coverage is true
    Returns:
        A `[batch_size, max_time]` tensor of unnormalized score values.
    """
    dtype = processed_query.dtype
    # Get the number of hidden units from the trailing dimension of keys
    num_units = keys.shape[2].value or tf.shape(keys)[2]
    # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
    processed_query = tf.expand_dims(processed_query, 1)
    v = tf.get_variable(
        "attention_v", [num_units], dtype=dtype)
    b = tf.get_variable(
        "attention_b", [num_units], dtype=dtype,
        initializer=tf.zeros_initializer())
    if coverage:
        w_c = tf.get_variable(
            "coverage_w", [num_units], dtype=dtype)
        # debug
        # coverage_vector = tf.Print(coverage_vector,[coverage_vector],message="score")
        coverage_vector = tf.expand_dims(coverage_vector, -1)
        return tf.reduce_sum(v * tf.tanh(keys + processed_query + coverage_vector * w_c + b), [2])
    else:
        return tf.reduce_sum(v * tf.tanh(keys + processed_query + b), [2])


class PGBahdanauAttention(seq2seq.BahdanauAttention):
    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 normalize=False,
                 probability_fn=None,
                 score_mask_value=float("-inf"),
                 name="PointerGeneratorBahdanauAttention",
                 coverage=False):
        """Construct the Attention mechanism.
        Args:
            num_units: The depth of the query mechanism.
            memory: The memory to query; usually the output of an RNN encoder.  This
            tensor should be shaped `[batch_size, max_time, ...]`.
            memory_sequence_length (optional): Sequence lengths for the batch entries
            in memory.  If provided, the memory tensor rows are masked with zeros
            for values past the respective sequence lengths.
            normalize: Python boolean.  Whether to normalize the energy term.
            probability_fn: (optional) A `callable`.  Converts the score to
            probabilities.  The default is @{tf.nn.softmax}. Other options include
            @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
            Its signature should be: `probabilities = probability_fn(score)`.
            score_mask_value: (optional): The mask value for score before passing into
            `probability_fn`. The default is -inf. Only used if
            `memory_sequence_length` is not None.
            name: Name to use when creating ops.
            coverage: whether use coverage mode
        """
        super(PGBahdanauAttention, self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            normalize=normalize,
            probability_fn=probability_fn,
            score_mask_value=score_mask_value,
            name=name)
        self.coverage = coverage

    def __call__(self, query, state):
        """Score the query based on the keys and values.
        Args:
            query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
            state: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).
        Returns:
            alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        """
        with tf.variable_scope(None, "pointer_generator_bahdanau_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            score = _pg_bahdanau_score(processed_query, self._keys, self.coverage, state)
        # Note: previous_alignments is not used in probability_fn in Bahda attention, so I use it as coverage vector in coverage mode
        alignments = self._probability_fn(score, state)

        return alignments, state
