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

    def build(self, x, x_len, q, q_len, y_begin, y_end, embeddings, keep_prob):
        """
        d: dim
        N: batch_size
        T: max_x
        J: max_q
        V: vocab_size
        """
        with tf.variable_scope('embedding_matrix'):
            # [V, d]
            emb_mat = tf.get_variable(name='emb_mat', shape=embeddings.shape, initializer=tf.constant_initializer(embeddings), trainable=False)
            tf.summary.histogram('emb_mat', emb_mat)
        
        with tf.variable_scope('embedding_context'):
            context = tf.nn.embedding_lookup(emb_mat, x, name='context') # X \in [N, T, d]

        with tf.variable_scope('embedding_question'):
            question = tf.nn.embedding_lookup(emb_mat, q, name='question') # Q \in [N, J, d]

        with tf.variable_scope('encoding_context'):
            if self.cell == 'gru':
                cell_x = GRUCell(self.dim)
            else:
                cell_x = LSTMCell(self.dim)
            cell_x = DropoutWrapper(cell_x, input_keep_prob=keep_prob)  # to avoid over-fitting

            outputs_context, _ = tf.nn.bidirectional_dynamic_rnn(cell_x, cell_x, inputs=context, sequence_length=x_len, dtype=tf.float32)
            context_fw, context_bw = outputs_context
            context_output = tf.concat([context_fw, context_bw], axis=2)  # H \in [N, T, 2d]
            tf.summary.histogram('context_output', context_output)

        with tf.variable_scope('encoding_question'):
            #tf.get_variable_scope().reuse_variables()
            if self.cell == 'gru':
                cell_q = GRUCell(self.dim)
            else:
                cell_q = LSTMCell(self.dim)
            cell_q = DropoutWrapper(cell_q, input_keep_prob=keep_prob)  # to avoid over-fitting

            outputs_question, _ = tf.nn.bidirectional_dynamic_rnn(cell_q, cell_q, inputs=question, sequence_length=q_len, dtype=tf.float32)
            question_fw, question_bw = outputs_question
            question_output = tf.concat([question_fw, question_bw], axis=2)   # U \in [N, J, 2d]
            tf.summary.histogram('question_output', question_output)

        q_mask = tf.sequence_mask(q_len, self.max_q)
        x_mask = tf.sequence_mask(x_len, self.max_x)

        with tf.variable_scope('shared_similarity_matrix'):
            x_exp = tf.tile(tf.expand_dims(context_output, axis=2), [1, 1, self.max_q, 1])  # [N, T, J, 2d]
            q_exp = tf.tile(tf.expand_dims(question_output, axis=1), [1, self.max_x, 1, 1])  # [N, T, J, 2d]
            inputs = tf.concat([x_exp, q_exp, x_exp * q_exp], 3)  # [N, T, J, 6d]
            mask = tf.tile(tf.expand_dims(x_mask, -1), [1, 1, self.max_q]) & tf.expand_dims(q_mask, 1)  # [N, T, J]
            val = tf.reshape(tf.layers.dense(inputs=inputs, units=1), [-1, self.max_x, self.max_q])
            logits = val - (1.0 - tf.cast(mask, 'float')) * 10.0e10  # S \in [N, T, J]
            
        with tf.variable_scope('context_to_query_attention'):
            probs = tf.nn.softmax(logits)
            sum_q = tf.reduce_sum(q_exp * tf.expand_dims(probs, -1), axis=2) # U^{tilde} \in [N, T, 2d]
            tf.summary.histogram('sum_q', sum_q)

        with tf.variable_scope('query_to_context_attention'):
            pass

        xq = tf.concat([context_output, sum_q, context_output * sum_q], axis=2)  # G \in [N, T, 6d]

        with tf.variable_scope('post_process_1'):
            if self.cell == 'gru':
                cell_xq_1 = GRUCell(self.dim)
            else:
                cell_xq_1 = LSTMCell(self.dim)
            cell_xq_1 = DropoutWrapper(cell_xq_1, input_keep_prob=keep_prob)  # to avoid over-fitting
            outputs_xq_1, _ = tf.nn.bidirectional_dynamic_rnn(cell_xq_1, cell_xq_1, inputs=xq, sequence_length=x_len, dtype=tf.float32)
            xq_fw_1, xq_bw_1 = outputs_xq_1
            xq_output_1 = tf.concat([xq_fw_1, xq_bw_1], axis=2)  # M \in [N, T, 2d]

        with tf.variable_scope('post_process_2'):
            if self.cell == 'gru':
                cell_xq_2 = GRUCell(self.dim)
            else:
                cell_xq_2 = LSTMCell(self.dim)
            cell_xq_2 = DropoutWrapper(cell_xq_2, input_keep_prob=keep_prob)  # to avoid over-fitting
            outputs_xq_2, _ = tf.nn.bidirectional_dynamic_rnn(cell_xq_2, cell_xq_2, inputs=xq_output_1, sequence_length=x_len, dtype=tf.float32)
            xq_fw_2, xq_bw_2 = outputs_xq_2
            xq_output_2 = tf.concat([xq_fw_2, xq_bw_2], axis=2)  # M \in [N, T, 2d]
            tf.summary.histogram('xq_output', xq_output_2)

        # Get rid of the sequence dimension
        xq_flat = tf.reshape(xq_output_2, [-1, 2 * self.dim])  # [N * T, 2d]

        # logits
        with tf.variable_scope('start_index'):
            val = tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, self.max_x])  # [N, T]
            logits_start = val - (1.0 - tf.cast(x_mask, 'float')) * 10.0e10
            yp_start = tf.argmax(logits_start, axis=1, name='starting_index')  # [N]
            tf.summary.histogram('yp_start', yp_start)

        with tf.variable_scope('end_index'):
            if self.cell == 'gru':
                cell_xq_end = GRUCell(self.dim)
            else:
                cell_xq_end = LSTMCell(self.dim)
            cell_xq_end = DropoutWrapper(cell_xq_end, input_keep_prob=keep_prob)  # to avoid over-fitting

            outputs_xq_end, _ = tf.nn.bidirectional_dynamic_rnn(cell_xq_end, cell_xq_end, inputs=xq, sequence_length=x_len, dtype=tf.float32)
            xq_fw_end, xq_bw_end = outputs_xq_end
            xq_output_end = tf.concat([xq_fw_end, xq_bw_end], axis=2)  # M^2 \in [N, T, 2d]
            xq_flat_end = tf.reshape(xq_output_end, [-1, 2 * self.dim])  # [N * T, 2d]
            val = tf.reshape(tf.layers.dense(inputs=xq_flat_end, units=1), [-1, self.max_x])  # [N, T]
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
