import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import GRUCell


class Model:
    def __init__(self, config, max_x, max_q):
        self.model_name = 'attention1'
        self.dim = config.hidden_size
        self.max_x = max_x
        self.max_q = max_q
        self.saver = None

    def build(self, x, x_len, q, q_len, embeddings, keep_prob):
        with tf.variable_scope('embedding_matrix'):
            # embeddings matrix, may be memory ineffecient (Fix)
            emb_mat = tf.get_variable(name="emb_mat", shape=embeddings.shape, initializer=tf.constant_initializer(embeddings), trainable=False)
            tf.summary.histogram("emb_mat", emb_mat)
        
        with tf.variable_scope('embedding_context'):
            context = tf.nn.embedding_lookup(emb_mat, x, name='context')

        with tf.variable_scope('embedding_question'):
            question = tf.nn.embedding_lookup(emb_mat, q, name='question')

        with tf.variable_scope('encoding_context'):
            gru_c_cell = GRUCell(self.dim)
            gru_c_cell = DropoutWrapper(gru_c_cell, input_keep_prob=keep_prob)  # to avoid over-fitting

            outputs_context, _ = tf.nn.bidirectional_dynamic_rnn(gru_c_cell, gru_c_cell, inputs=context, sequence_length=x_len, dtype=tf.float32)
            context_fw, context_bw = outputs_context
            context_output = tf.concat([context_fw, context_bw], axis=2)
            tf.summary.histogram("context_output", context_output)

        with tf.variable_scope('encoding_question'):
            gru_q_cell = GRUCell(self.dim)
            gru_q_cell = DropoutWrapper(gru_q_cell, input_keep_prob=keep_prob)  # to avoid over-fitting

            outputs_question, _ = tf.nn.bidirectional_dynamic_rnn(gru_q_cell, gru_q_cell, inputs=question, sequence_length=q_len, dtype=tf.float32)
            question_fw, question_bw = outputs_question
            question_output = tf.concat([question_fw, question_bw], axis=2)
            tf.summary.histogram("question_output", question_output)

        q_mask = tf.sequence_mask(q_len, self.max_q)
        x_mask = tf.sequence_mask(x_len, self.max_x)

        with tf.variable_scope("attention"):
            x_exp = tf.tile(tf.expand_dims(context_output, axis=2), [1, 1, self.max_q, 1])
            q_exp = tf.tile(tf.expand_dims(question_output, axis=1), [1, self.max_x, 1, 1])
            inputs = tf.concat([x_exp, q_exp, x_exp * q_exp], 3)
            mask = tf.tile(tf.expand_dims(x_mask, -1), [1, 1, self.max_q]) & tf.expand_dims(q_mask, 1)
            val = tf.reshape(tf.layers.dense(inputs=inputs, units=1), [-1, self.max_x, self.max_q])
            logits = val - (1.0 - tf.cast(mask, 'float')) * 10.0e10
            probs = tf.nn.softmax(logits)
            sum_q = tf.reduce_sum(q_exp * tf.expand_dims(probs, -1), axis=2)
            tf.summary.histogram("sum_q", sum_q)

        xq = tf.concat([context_output, sum_q, context_output * sum_q], axis=2)

        with tf.variable_scope('post_process'):
            gru_xq_cell = GRUCell(self.dim)
            gru_xq_cell = DropoutWrapper(gru_xq_cell, input_keep_prob=keep_prob)  # to avoid over-fitting

            outputs_xq, _ = tf.nn.bidirectional_dynamic_rnn(gru_xq_cell, gru_xq_cell, inputs=xq, sequence_length=x_len, dtype=tf.float32)
            xq_fw, xq_bw = outputs_xq
            xq_output = tf.concat([xq_fw, xq_bw], axis=2)
            tf.summary.histogram("xq_output", xq_output)

        # Get rid of the sequence dimension
        xq_flat = tf.reshape(xq_output, [-1, 2 * self.dim])

        # tensor of boolean values of max_x length and True in first x_len indices
        x_mask = tf.sequence_mask(x_len, self.max_x)

        # logits
        with tf.variable_scope('start_index'):
            val = tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, self.max_x])
            logits_start = val - (1.0 - tf.cast(x_mask, 'float')) * 10.0e10
            yp_start = tf.argmax(logits_start, axis=1, name='starting_index')
            tf.summary.histogram("yp_start", yp_start)

        with tf.variable_scope('end_index'):
            val = tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, self.max_x])
            logits_end = val - (1.0 - tf.cast(x_mask, 'float')) * 10.0e10
            yp_end = tf.argmax(logits_end, axis=1, name='ending_index')
            tf.summary.histogram("yp_end", yp_end)

        outputs = {'logits_start': logits_start, 'logits_end': logits_end, 'yp_start': yp_start, 'yp_end': yp_end}
        return outputs
