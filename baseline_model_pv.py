import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import GRUCell


class Model:
    def __init__(self, config, max_x, max_q):
        self.model_name = 'baseline'
        self.dim = config.hidden_size
        self.max_x = max_x
        self.max_q = max_q
        self.saver = None

    def build(self, xs, x_len, q, q_len, y, embeddings, keep_prob):
        with tf.variable_scope('embedding_matrix'):
            # embeddings matrix, may be memory ineffecient (Fix)
            emb_mat = tf.get_variable(name='emb_mat', shape=embeddings.shape, initializer=tf.constant_initializer(embeddings), trainable=False)
            tf.summary.histogram('emb_mat', emb_mat)

        with tf.variable_scope('embedding_passages'):
            context = []
            
            for x in xs:
                context.append(tf.nn.embedding_lookup(emb_mat, x, name='context'))

        with tf.variable_scope('embedding_question'):
            question = tf.nn.embedding_lookup(emb_mat, q, name='question')

        with tf.variable_scope('encoding_passages') as scope:
            gru_c_cell = GRUCell(self.dim)
            gru_c_cell = DropoutWrapper(gru_c_cell, input_keep_prob=keep_prob)  # to avoid over-fitting

            context_output = []
            for i in range(len(context)):
                if i > 0:
                    scope.reuse_variables()
                print(context[i].get_shape())
                print(x_len.get_shape())
                outputs_context, _ = tf.nn.bidirectional_dynamic_rnn(gru_c_cell, gru_c_cell, inputs=context[i], sequence_length=x_len, dtype=tf.float32)
                context_fw, context_bw = outputs_context
                context_output.append(tf.concat([context_fw, context_bw], axis=2))
                # tf.summary.histogram('context_output', context_output)
            context_output = tf.concat(context_output, axis=2)

        with tf.variable_scope('encoding_question'):
            gru_q_cell = GRUCell(self.dim)
            gru_q_cell = DropoutWrapper(gru_q_cell, input_keep_prob=keep_prob)  # to avoid over-fitting

            # tf.get_variable_scope().reuse_variables()
            outputs_question, _ = tf.nn.bidirectional_dynamic_rnn(gru_q_cell, gru_q_cell, inputs=question, sequence_length=q_len, dtype=tf.float32)
            question_fw, question_bw = outputs_question
            question_output = tf.concat([question_fw, question_bw], axis=2)
            tf.summary.histogram('question_output', question_output)

        with tf.variable_scope('question_tiling'):
            q_mask = tf.sequence_mask(q_len, self.max_q)
            mask = tf.expand_dims(q_mask, -1)
            q_temp = question_output * tf.cast(mask, 'float')
            q_avg = tf.reduce_mean(q_temp, axis=1)
            q_avg_exp = tf.expand_dims(q_avg, axis=1)
            q_avg_tiled = tf.tile(q_avg_exp, [1, self.max_x, 1])
            tf.summary.histogram('q_avg_tiled', q_avg)

        xq = tf.concat([context_output, q_avg_tiled], axis=2)
        tf.summary.histogram('xq', xq)

        with tf.variable_scope('post_process'):
            gru_xq_cell = GRUCell(self.dim)
            gru_xq_cell = DropoutWrapper(gru_xq_cell, input_keep_prob=keep_prob)  # to avoid over-fitting

            outputs_xq, _ = tf.nn.bidirectional_dynamic_rnn(gru_xq_cell, gru_xq_cell, inputs=xq, sequence_length=x_len, dtype=tf.float32)
            xq_fw, xq_bw = outputs_xq
            xq_output = tf.concat([xq_fw, xq_bw], axis=2)
            tf.summary.histogram('xq_output', xq_output)

        # logits
        with tf.variable_scope('relevance'):
            logits = tf.layers.dense(inputs=xq_output, units=10)

        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits), name='loss')

        tf.summary.scalar('loss', loss)

        self.loss = loss
        self.merged_summary = tf.summary.merge_all()
