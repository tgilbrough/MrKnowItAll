import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import GRUCell


class Model:
    def __init__(self, config, max_x, max_q):
        self.model_name = 'bidaf'
        self.dim = config.hidden_size
        self.max_x = max_x
        self.max_q = max_q
        self.saver = None
        self.cell = config.cell
        self.highway_network_use = config.highway_network
        self.batch_size = config.batch_size

    def build(self, x, x_len, q, q_len, y_begin, y_end, embeddings, keep_prob):
        """
        d: dim
        N: batch_size
        MX: max_x
        MQ: max_q
        V: vocab_size
        """
        with tf.variable_scope('embedding_matrix'):
            # [V, d]
            emb_mat = tf.get_variable(name='emb_mat', shape=embeddings.shape, initializer=tf.constant_initializer(embeddings), trainable=False)
        
        with tf.variable_scope('embedding_context'):
            # [N, MX, d]
            context = tf.nn.embedding_lookup(emb_mat, x, name='context')

        with tf.variable_scope('embedding_question'):
            # [N, MQ, d]
            question = tf.nn.embedding_lookup(emb_mat, q, name='question')

        with tf.variable_scope('highway_network'):
            if self.highway_network_use:
                layers = 2
                carry_bias = -1.0
                context = self.highway_network(context, layers, carry_bias)
                tf.get_variable_scope().reuse_variables()
                question = self.highway_network(question, layers, carry_bias)
                tf.summary.histogram('context', context)
                tf.summary.histogram('question', question)

        if self.cell == 'gru':
            cell = GRUCell(self.dim)
        else:
            cell = LSTMCell(self.dim)
        d_cell = DropoutWrapper(cell, input_keep_prob=keep_prob)  # to avoid over-fitting

        with tf.variable_scope('encoding'):
            outputs_question, _ = tf.nn.bidirectional_dynamic_rnn(d_cell, d_cell, inputs=question, sequence_length=q_len, dtype=tf.float32)
            question_fw, question_bw = outputs_question
            question_output = tf.concat([question_fw, question_bw], axis=2)  # [N, MQ, 2d]
            tf.summary.histogram('question_output', question_output)

            tf.get_variable_scope().reuse_variables()

            outputs_context, _ = tf.nn.bidirectional_dynamic_rnn(d_cell, d_cell, inputs=context, sequence_length=x_len, dtype=tf.float32)
            context_fw, context_bw = outputs_context
            context_output = tf.concat([context_fw, context_bw], axis=2)  # [N, MX, 2d]
            tf.summary.histogram('context_output', context_output)

        q_mask = tf.sequence_mask(q_len, self.max_q)
        x_mask = tf.sequence_mask(x_len, self.max_x)

        with tf.variable_scope('attention'):
            x_exp = tf.tile(tf.expand_dims(context_output, axis=2), [1, 1, self.max_q, 1])  # [N, MX, MQ, 2d]
            q_exp = tf.tile(tf.expand_dims(question_output, axis=1), [1, self.max_x, 1, 1])  # [N, MX, MQ, 2d]
            inputs = tf.concat([x_exp, q_exp, x_exp * q_exp], 3)  # [N, MX, MQ, 6d]
            val = tf.reshape(tf.layers.dense(inputs=inputs, units=1), [-1, self.max_x, self.max_q])

        with tf.variable_scope('contex_to_query_attention'):
            mask = tf.tile(tf.expand_dims(x_mask, -1), [1, 1, self.max_q]) & tf.expand_dims(q_mask, 1)  # [N, MX, MQ]
            logits = val - (1.0 - tf.cast(mask, 'float')) * 10.0e10  # [N, MX, MQ]
            probs = tf.nn.softmax(logits)
            sum_q = tf.reduce_sum(q_exp * tf.expand_dims(probs, -1), axis=2)
            tf.summary.histogram('sum_q', sum_q)

        with tf.variable_scope('query_to_context_attention'):
            mx_1 = tf.reduce_max(logits, axis=2)  # [N, MX]
            mx_2 = tf.nn.softmax(mx_1)  # [N, MX]
            mx_3 = tf.expand_dims(tf.expand_dims(mx_2, -1), -1)
            sum_x = tf.reduce_sum(x_exp * mx_3, axis=2)

        xq = tf.concat([context_output, sum_q, context_output * sum_q, context_output * sum_x], axis=2)

        with tf.variable_scope('post_process_1'):
            outputs_xq_1, _ = tf.nn.bidirectional_dynamic_rnn(d_cell, d_cell, inputs=xq, sequence_length=x_len, dtype=tf.float32)
            xq_fw_1, xq_bw_1 = outputs_xq_1
            xq_output_1 = tf.concat([xq_fw_1, xq_bw_1], axis=2)  # [N, MX, 2d]

        with tf.variable_scope('post_process_2'):
            outputs_xq_2, _ = tf.nn.bidirectional_dynamic_rnn(d_cell, d_cell, inputs=xq_output_1, sequence_length=x_len, dtype=tf.float32)
            xq_fw_2, xq_bw_2 = outputs_xq_2
            xq_output_2 = tf.concat([xq_fw_2, xq_bw_2], axis=2)  # [N, MX, 2d]
            tf.summary.histogram('xq_output', xq_output_2)

        with tf.variable_scope('start_index'):
            # Get rid of the sequence dimension
            xq_flat = tf.reshape(xq_output_2, [-1, 2 * self.dim])  # [N * MX, 2d]

            val = tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, self.max_x])  # [N, MX]
            logits_start = val - (1.0 - tf.cast(x_mask, 'float')) * 10.0e10
            yp_start = tf.argmax(logits_start, axis=1, name='starting_index')  # [N]
            tf.summary.histogram('yp_start', yp_start)

        with tf.variable_scope('end_index'):
            a1i = tf.tile(tf.expand_dims(logits_start, 2), [1, 1, 2 * self.dim])
            inputs = tf.concat([xq, xq_output_2, a1i, xq_output_2 * a1i], axis=2)
            outputs_xq_end, _ = tf.nn.bidirectional_dynamic_rnn(d_cell, d_cell, inputs=inputs, sequence_length=x_len, dtype=tf.float32)
            xq_fw_end, xq_bw_end = outputs_xq_end
            
            xq_output_end = tf.concat([xq_fw_end, xq_bw_end], axis=2)  # [N, MX, 2d]

            xq_flat_end = tf.reshape(xq_output_end, [-1, 2 * self.dim])  # [N * MX, 2d]
            val = tf.reshape(tf.layers.dense(inputs=xq_flat_end, units=1), [-1, self.max_x])  # [N, MX]
            logits_end = val - (1.0 - tf.cast(x_mask, 'float')) * 10.0e10
            yp_end = tf.argmax(logits_end, axis=1, name='ending_index')  # [N]
            tf.summary.histogram('yp_end', yp_end)

        with tf.variable_scope('loss'):
            loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_begin, logits=logits_start), name='beginning_loss')
            loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_end, logits=logits_end), name='ending_loss')
            loss = loss1 + loss2
        with tf.variable_scope('accuracy'):
            acc1 = tf.reduce_mean(tf.cast(tf.equal(y_begin, tf.cast(tf.argmax(logits_start, 1), 'int32')), 'float'), name='beginning_accuracy')
            acc2 = tf.reduce_mean(tf.cast(tf.equal(y_end, tf.cast(tf.argmax(logits_end, 1), 'int32')), 'float'), name='ending_accuracy')

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('loss1', loss1)
        tf.summary.scalar('loss2', loss2)
        tf.summary.scalar('accuracy1', acc1)
        tf.summary.scalar('accuracy2', acc2)

        self.logits1 = logits_start
        self.logits2 = logits_end

        self.loss = loss
        self.merged_summary = tf.summary.merge_all()

    def highway_network(self, x, layers, carry_bias):
        prev = x
        curr = None
        for i in range(layers):
            curr = self.highway_layer(prev, carry_bias)
            prev = curr
        return curr

    def highway_layer(self, x, carry_bias):
        W_T = tf.Variable(tf.truncated_normal([self.dim, self.dim], stddev=0.1), name="weight_transform")
        b_T = tf.Variable(tf.constant(carry_bias, shape=[self.dim]), name="bias_transform")

        W = tf.Variable(tf.truncated_normal([self.dim, self.dim], stddev=0.1), name="weight")
        b = tf.Variable(tf.constant(0.1, shape=[self.dim]), name="bias")

        H = tf.nn.softmax(self.batch_matmul(x, W) + b, name="activation")
        T = tf.sigmoid(self.batch_matmul(x, W_T) + b_T, name="transform_gate")
        C = tf.subtract(1.0, T, name="carry_gate")

        y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")
        return y

    def batch_matmul(self, x, W):
        shape = x.get_shape()
        m = shape[1].value
        n = shape[2].value

        c = W.get_shape()[0].value

        x = tf.reshape(x, [-1, n])
        y = tf.matmul(x, W)
        y = tf.reshape(y, [-1, m, c])
        return y
