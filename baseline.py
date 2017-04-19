import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import GRUCell


class Model:
    def __init__(self, config, max_x, max_q):
        self.model_name = 'baseline'
        self.dim = config.hidden_size
        self.max_x = max_x
        self.max_q = max_q

    def build(self, x, x_len, q, q_len, embeddings, keep_prob):
        with tf.variable_scope('embedding_matrix'):
            # embeddings matrix, may be memory ineffecient (Fix)
            emb_mat = tf.get_variable(name="emb_mat", shape=embeddings.shape, initializer=tf.constant_initializer(embeddings), trainable=False)

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

        with tf.variable_scope('encoding_question'):
            gru_q_cell = GRUCell(self.dim)
            gru_q_cell = DropoutWrapper(gru_q_cell, input_keep_prob=keep_prob)  # to avoid over-fitting

            # tf.get_variable_scope().reuse_variables()
            outputs_question, _ = tf.nn.bidirectional_dynamic_rnn(gru_q_cell, gru_q_cell, inputs=question, sequence_length=q_len, dtype=tf.float32)
            question_fw, question_bw = outputs_question
            question_output = tf.concat([question_fw, question_bw], axis=2)

        with tf.variable_scope('question_tiling'):
            q_mask = tf.sequence_mask(q_len, self.max_q)
            mask = tf.expand_dims(q_mask, -1)
            q_temp = question_output * tf.cast(mask, 'float')
            q_avg = tf.reduce_mean(q_temp, axis=1)
            q_avg_exp = tf.expand_dims(q_avg, axis=1)
            q_avg_tiled = tf.tile(q_avg_exp, [1, self.max_x, 1])

        xq = tf.concat([context_output, q_avg_tiled, context_output * q_avg_tiled], axis=2)

        with tf.variable_scope('post_process'):
            gru_xq_cell = GRUCell(self.dim)
            gru_xq_cell = DropoutWrapper(gru_xq_cell, input_keep_prob=keep_prob)  # to avoid over-fitting

            outputs_xq, _ = tf.nn.bidirectional_dynamic_rnn(gru_xq_cell, gru_xq_cell, inputs=xq, sequence_length=x_len, dtype=tf.float32)
            xq_fw, xq_bw = outputs_xq
            xq_output = tf.concat([xq_fw, xq_bw], axis=2)

        # Get rid of the sequence dimension
        xq_flat = tf.reshape(xq_output, [-1, 2 * self.dim])

        # tensor of boolean values of max_x length and True in first x_len indices
        x_mask = tf.sequence_mask(x_len, self.max_x)

        # logits
        with tf.variable_scope('start_index'):
            val = tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, self.max_x])
            logits_start = val - (1.0 - tf.cast(x_mask, 'float')) * 10.0e10
            yp_start = tf.argmax(logits_start, axis=1, name='starting_index')

        with tf.variable_scope('end_index'):
            val = tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, self.max_x])
            logits_end = val - (1.0 - tf.cast(x_mask, 'float')) * 10.0e10
            yp_end = tf.argmax(logits_end, axis=1, name='ending_index')

        outputs = {'logits_start': logits_start, 'logits_end': logits_end, 'yp_start': yp_start, 'yp_end': yp_end}
        return outputs

    def loss(self, outputs):
        # Place holder for just index of answer within context  
        y_begin = tf.placeholder(tf.int32, [None], name='y_begin')
        y_end = tf.placeholder(tf.int32, [None], name='y_end')

        logits1, logits2 = outputs['logits_start'], outputs['logits_end']
        loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_begin, logits=logits1))
        loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_end, logits=logits2))
        loss = loss1 + loss2
        acc1 = tf.reduce_mean(tf.cast(tf.equal(y_begin, tf.cast(tf.argmax(logits1, 1), 'int32')), 'float'))
        acc2 = tf.reduce_mean(tf.cast(tf.equal(y_end, tf.cast(tf.argmax(logits2, 1), 'int32')), 'float'))

        loss_metrics = {'loss': loss, 'acc1': acc1, 'acc2': acc2}
        return loss_metrics
