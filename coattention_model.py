import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import GRUCell, LSTMCell
import numpy as np
from tf_ops import highway_maxout, batch_linear

# Attempt at recreating: https://arxiv.org/pdf/1611.01604.pdf
class Model:
    def __init__(self, config, max_x, max_q):
        self.model_name = 'coattention'
        self.emb_size = config.emb_size
        self.hidden_size = config.hidden_size
        self.pool_size = config.pool_size
        self.max_decode_steps = config.max_decode_steps
        self.max_x = max_x
        self.max_q = max_q
        self.saver = None

    def build(self, x, x_len, q, q_len, y_begin, y_end, embeddings, keep_prob):
        with tf.variable_scope('embedding_matrix'):
            # embeddings matrix, may be memory ineffecient (Fix)
            emb_mat = tf.get_variable(name='emb_mat', shape=embeddings.shape, initializer=tf.constant_initializer(embeddings), trainable=False)
            tf.summary.histogram('emb_mat', emb_mat)
        
        with tf.variable_scope('embedding'):
            context = tf.nn.embedding_lookup(emb_mat, x, name='context') # (batch_size, max_x, emb_size)
            question = tf.nn.embedding_lookup(emb_mat, q, name='question') # (batch_size, max_q, emb_size)

        with tf.variable_scope('encoding') as scope:
            lstm_enc = LSTMCell(self.hidden_size) 
            lstm_enc = DropoutWrapper(lstm_enc, input_keep_prob=keep_prob)
        
            # Add sentinel to end of encodings
            sentinel = tf.get_variable('sentinel', [1, self.hidden_size], dtype=tf.float32)
            fn = lambda x: tf.concat([x, sentinel], axis=0)

            D, _ = tf.nn.dynamic_rnn(lstm_enc, context, sequence_length=x_len, dtype=tf.float32) # (batch_size, max_x, hidden_size)
            D = tf.map_fn(lambda x: fn(x), D, dtype=tf.float32)
            D = tf.transpose(D, perm=[0, 2, 1]) # (batch_size, hidden_size, max_x)            
            tf.summary.histogram('D', D)

            scope.reuse_variables()

            q, _ = tf.nn.dynamic_rnn(lstm_enc, question, sequence_length=q_len, dtype=tf.float32) # (batch_size, max_q, hidden_size)
            q = tf.map_fn(lambda x: fn(x), q, dtype=tf.float32)
            q = tf.transpose(q, perm=[0, 2, 1]) # (batch_size, hidden_size, max_q)
            tf.summary.histogram('Q_', q)

        with tf.variable_scope('transforming_question'):
            Q = q 

        with tf.variable_scope('affinity_mat'):
            L = tf.matmul(D, Q, name='L', transpose_a=True) # (batch_size, max_x, max_q)
            tf.summary.histogram('L', L)

        with tf.variable_scope('normalize_aff'):
            Aq = tf.nn.softmax(L, name='Aq') # (batch_size, max_x, max_q)
            Ad = tf.nn.softmax(tf.transpose(L, perm=[0, 2, 1]), name='Ad') # (batch_size, max_q, max_x)
            tf.summary.histogram('Aq', Aq)
            tf.summary.histogram('Ad', Ad)

        with tf.variable_scope('attention_contexts'):
            Cq = tf.matmul(D, Aq, name='Cq') # (batch_size, hidden_size, max_q)
            tf.summary.histogram('Cq', Cq)

        with tf.variable_scope('attention_questions'):
            Cd = tf.concat([Q, Cq], axis=1) # (batch_size, 2*hidden_size, max_q)
            Cd = tf.matmul(Cd, Ad, name='Cd') # (batch_size, 2*hidden_size, max_x)
            tf.summary.histogram('Cd', Cd)
        
        with tf.variable_scope('encoding_understanding'):
            lstm_fw_cell = LSTMCell(self.hidden_size)
            lstm_bw_cell = LSTMCell(self.hidden_size)
            lstm_fw_cell = DropoutWrapper(lstm_fw_cell, input_keep_prob=keep_prob)
            lstm_bw_cell = DropoutWrapper(lstm_bw_cell, input_keep_prob=keep_prob)

            inputs_ = tf.concat([D, Cd], axis=1) # (batch_size, 3*hidden_size, max_x)
            inputs_ = tf.transpose(inputs_, perm=[0, 2, 1]) # (batch_size, max_x, 3*hidden_size)

            u, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs=inputs_, sequence_length=x_len, dtype=tf.float32) # (batch_size, max_x, hidden_size)
    
            U = tf.concat(u, axis=2) # (batch_size, max_x, 2*hidden_size)
            tf.summary.histogram('U', U)

        def batch_gather(a, b):
            b_2 = tf.expand_dims(b, 1)
            range_ = tf.expand_dims(tf.range(tf.shape(b)[0]), 1)
            ind = tf.concat([range_, b_2], axis=1)
            return tf.gather_nd(a, ind) 

        with tf.variable_scope('selector'):        
            batch_size = tf.shape(U)[0]

            U_trans = tf.transpose(U, perm=[1, 0, 2]) # (max_q, batch_size, 2*hidden_size) //the u_t vectors

            # batch indices
            loop_until = tf.range(0, batch_size, dtype=tf.int32)

            # initial estimated positions
            s = tf.zeros([batch_size], dtype=tf.int32, name='s') # (batch_size)
            e = tf.zeros([batch_size], dtype=tf.int32, name='e') # (batch_size)

            alpha = tf.layers.dense(U, 1, name='s')
            alpha = tf.reshape(alpha, [-1, self.max_x + 1])
            s = tf.reshape(tf.cast(tf.argmax(alpha, axis=1), tf.int32), [batch_size]) 


            beta = tf.layers.dense(U, 1, name='e')
            beta = tf.reshape(beta, [-1, self.max_x + 1])
            e = tf.reshape(tf.cast(tf.argmax(beta, axis=1), tf.int32), [batch_size])


            # Get U vectors of starting indexes
            u_s = batch_gather(U, s) # (batch_size, 2*hidden_size)

            # Get U vectors of ending indexes
            u_e = batch_gather(U, e) # (batch_size, 2*hidden_size)

        with tf.variable_scope('highway_init'):
            highway_alpha = highway_maxout(self.hidden_size, self.pool_size)
            highway_beta = highway_maxout(self.hidden_size, self.pool_size)

        self._s, self._e = [], []
        self._alpha, self._beta = [alpha], [beta]
        with tf.variable_scope('decoder') as scope:
            # LSTM for decoding
            lstm_dec = LSTMCell(self.hidden_size)
            lstm_dec = DropoutWrapper(lstm_dec, input_keep_prob=keep_prob)

            for step in range(self.max_decode_steps):
                if step > 0:
                    scope.reuse_variables()
                
                _input = tf.concat([u_s, u_e], axis=1) # (batch_size, 4*hidden_size)

                # single step lstm
                h,_ = tf.contrib.rnn.static_rnn(lstm_dec, [_input], dtype=tf.float32) # (batch_size, hidden_size)

                h_state = h[0] # h_state = tf.concat(h, axis=1)
                
                tf.summary.histogram('h_state', h_state)

                with tf.variable_scope('highway_alpha'):
                    # compute start position first
                    fn = lambda u_t: highway_alpha(u_t, h_state, u_s, u_e)
                    
                    # for each t, send in (batch_size, hidden_size) matrix 
                    alpha = tf.map_fn(fn, U_trans, dtype=tf.float32) # (max_x, batch_size, 1, 1)
                    tf.summary.histogram('alpha_iter_' + str(step + 1), alpha)

                    s = tf.reshape(tf.cast(tf.argmax(alpha, axis=0), tf.int32), [batch_size]) # (batch_size)

                    # update start guess
                    u_s = batch_gather(U, s) # (batch_size, 2*hidden_size)

                with tf.variable_scope('highway_beta'):
                    # compute end position next
                    fn = lambda u_t: highway_beta(u_t, h_state, u_s, u_e)
                    
                    beta = tf.map_fn(fn, U_trans, dtype=tf.float32) # (max_x, batch_size, 1, 1)
                    tf.summary.histogram('beta_iter_' + str(step + 1), beta)

                    e = tf.reshape(tf.cast(tf.argmax(beta, axis=0), tf.int32), [batch_size]) # (batch_size)
                    
                    # update end guess
                    u_e = batch_gather(U, e) # (batch_size, 2*hidden_size)

                self._s.append(s)
                self._e.append(e)
                self._alpha.append(tf.reshape(alpha, [batch_size, -1]))
                self._beta.append(tf.reshape(beta, [batch_size, -1]))   

        self.loss = self._loss_multitask(self._alpha, y_begin, self._beta, y_end)

        tf.summary.scalar('loss', self.loss)

        self.logits1 = self._alpha[-1]
        self.logits2 = self._beta[-1]
        
        self.merged_summary = tf.summary.merge_all()
    

    def _loss_multitask(self, logits_alpha, labels_alpha, logits_beta, labels_beta):
        '''Cumulative loss for start and end positions.'''
        fn = lambda logit, label: self._loss_shared(logit, label)
        loss_alpha = [fn(alpha, labels_alpha) for alpha in logits_alpha]
        loss_beta = [fn(beta, labels_beta) for beta in logits_beta]
        return tf.reduce_sum([loss_alpha, loss_beta], name='loss')

    def _loss_shared(self, logits, labels):
        labels = tf.reshape(labels, [-1])
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='per_step_cross_entropy')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('per_step_losses', cross_entropy_mean)
        return tf.add_n(tf.get_collection('per_step_losses'), name='per_step_loss')