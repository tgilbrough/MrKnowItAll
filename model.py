import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import GRUCell


class Model:
    def __init__(self, config):
        self.keep_prob = config.keep_prob
        self.dim = config.hidden_size
        self.max_x = config.max_context_size
        self.max_q = config.max_ques_size

    def compute(self, x, x_len, q, q_len, embeddings):
        # embeddings matrix, may be memory ineffecient (Fix)
        emb_mat = tf.get_variable(name="emb_mat", shape=embeddings.shape, initializer=tf.constant_initializer(embeddings), trainable=False)
        
        # tensor of boolean values of max_x length and True in first x_len indices
        x_mask = tf.sequence_mask(x_len, self.max_x)
        q_mask = tf.sequence_mask(q_len, self.max_q)

        context = tf.nn.embedding_lookup(emb_mat, x, name='context')
        question = tf.nn.embedding_lookup(emb_mat, q, name='question')

        with tf.variable_scope('pre_process'):
            gru_c_fw_cell = GRUCell(self.dim)
            gru_c_fw_cell = DropoutWrapper(gru_c_fw_cell, input_keep_prob=self.keep_prob)  # to avoid over-fitting

            gru_c_bw_cell = GRUCell(self.dim)
            gru_c_bw_cell = DropoutWrapper(gru_c_bw_cell, input_keep_prob=self.keep_prob)  # to avoid over-fitting

            outputs_context, _ = tf.nn.bidirectional_dynamic_rnn(gru_c_fw_cell, gru_c_bw_cell, inputs=context, sequence_length=x_len, dtype=tf.float32)
            context_fw, context_bw = outputs_context
            context_output = tf.concat([context_fw, context_bw], axis=2)

            gru_q_fw_cell = GRUCell(self.dim)
            gru_q_fw_cell = DropoutWrapper(gru_q_fw_cell, input_keep_prob=self.keep_prob)  # to avoid over-fitting

            gru_q_bw_cell = GRUCell(self.dim)
            gru_q_bw_cell = DropoutWrapper(gru_q_bw_cell, input_keep_prob=self.keep_prob)  # to avoid over-fitting

            tf.get_variable_scope().reuse_variables()
            outputs_question, _ = tf.nn.bidirectional_dynamic_rnn(gru_q_fw_cell, gru_q_bw_cell, inputs=question, sequence_length=q_len, dtype=tf.float32)
            question_fw, question_bw = outputs_question
            question_output = tf.concat([question_fw, question_bw], axis=2)

        mask = tf.expand_dims(q_mask, -1)
        q_temp = question_output * tf.cast(mask, 'float')
        q_avg = tf.reduce_mean(q_temp, axis=1)
        q_avg_exp = tf.expand_dims(q_avg, axis=1)
        q_avg_tiled = tf.tile(q_avg_exp, [1, self.max_x, 1])

        xq = tf.concat([context_output, q_avg_tiled, context_output * q_avg_tiled], axis=2)

        with tf.variable_scope("post_process"):
            gru_xq_fw_cell = GRUCell(self.dim)
            gru_xq_fw_cell = DropoutWrapper(gru_xq_fw_cell, input_keep_prob=self.keep_prob)  # to avoid over-fitting

            gru_xq_bw_cell = GRUCell(self.dim)
            gru_xq_bw_cell = DropoutWrapper(gru_xq_bw_cell, input_keep_prob=self.keep_prob)  # to avoid over-fitting

            outputs_xq, _ = tf.nn.bidirectional_dynamic_rnn(gru_xq_fw_cell, gru_xq_bw_cell, inputs=xq, sequence_length=x_len, dtype=tf.float32)
            xq_fw, xq_bw = outputs_xq
            xq_output = tf.concat([xq_fw, xq_bw], axis=2)

        xq_flat = tf.reshape(xq_output, [-1, 2 * self.dim])

        # logits
        with tf.variable_scope('start'):
            val = tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, self.max_x])
            logits_start = val - (1.0 - tf.cast(x_mask, 'float')) * 10.0e10
            yp_start = tf.argmax(logits_start, axis=1)

        with tf.variable_scope('end'):
            val = tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, self.max_x])
            logits_end = val - (1.0 - tf.cast(x_mask, 'float')) * 10.0e10
            yp_end = tf.argmax(logits_end, axis=1)

        outputs = {'logits_start': logits_start, 'logits_end': logits_end, 'yp_start': yp_start, 'yp_end': yp_end}
        return outputs
