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

    def build(self, x, x_len, q, q_len, embeddings, keep_prob):
        with tf.variable_scope('embedding_matrix'):
            # embeddings matrix, may be memory ineffecient (Fix)
            emb_mat = tf.get_variable(name="emb_mat", shape=embeddings.shape, initializer=tf.constant_initializer(embeddings), trainable=False)
            tf.summary.histogram("emb_mat", emb_mat)
        
        with tf.variable_scope('embedding_context'):
            context = tf.nn.embedding_lookup(emb_mat, x, name='context')

        with tf.variable_scope('embedding_question'):
            question = tf.nn.embedding_lookup(emb_mat, q, name='question')

        lstm_enc = LSTMCell(self.hidden_size)
        lstm_enc = DropoutWrapper(lstm_enc, input_keep_prob=keep_prob)
        
        with tf.variable_scope('encoding_context'):
            D, _ = tf.nn.dynamic_rnn(lstm_enc, context, sequence_length=x_len, dtype=tf.float32)
            D = tf.transpose(D, perm=[0, 2, 1]) 
            tf.summary.histogram("D", D)

        with tf.variable_scope('encoding_question'):
            q, _ = tf.nn.dynamic_rnn(lstm_enc, question, sequence_length=q_len, dtype=tf.float32)
            q = tf.transpose(q, perm=[0, 2, 1]) 
            tf.summary.histogram("Q'", q)

            Q = tf.tanh(batch_linear(q, self.hidden_size, True))
            tf.summary.histogram("Q", Q)   
            
        print('D:', D.get_shape())
        print('Q:', Q.get_shape())

        with tf.variable_scope('affinity_mat'):
            L = tf.matmul(tf.transpose(D, perm=[0, 2, 1]), Q, name='L')

        print('L:', L.get_shape())

        with tf.variable_scope('normalize_aff'):
            Aq = tf.nn.softmax(L, name='Aq')
            Ad = tf.nn.softmax(tf.transpose(L, perm=[0, 2, 1]), name='Ad')

        print('Aq:', Aq.get_shape())
        print('Ad:', Ad.get_shape())

        with tf.variable_scope('attention_contexts'):
            Cq = tf.matmul(D, Aq, name='Cq')

        print('Cq:', Cq.get_shape())

        with tf.variable_scope('attention_questions'):
            Cd = tf.concat([Q, Cq], axis=1)
            Cd = tf.matmul(Cd, Ad, name='Cd')
        
        print('Cd:', Cd.get_shape())

        co_att = tf.concat([D, Cd], axis=1)
        co_att = tf.transpose(co_att, perm=[0, 2, 1])

        print('co_att:', co_att.get_shape())
        
        with tf.variable_scope('encoding_understanding'):
            lstm_fw_cell = LSTMCell(self.hidden_size)
            lstm_bw_cell = LSTMCell(self.hidden_size)
            lstm_fw_cell = DropoutWrapper(lstm_fw_cell, input_keep_prob=keep_prob)
            lstm_bw_cell = DropoutWrapper(lstm_bw_cell, input_keep_prob=keep_prob)

            u, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs=co_att, sequence_length=x_len, dtype=tf.float32)
    
            U = tf.concat(u, axis=2)
            U = tf.transpose(U, perm=[0, 2, 1])
        
        print('U:', U.get_shape())

        def select(u, pos, idx):
            u_idx = tf.gather(u, idx) # U for batch idx
            pos_idx = tf.gather(pos, idx) # Start for batch idx
            return tf.reshape(tf.gather(u_idx, pos_idx), [-1])

        with tf.variable_scope('selector'):        
            
            batch_size = tf.shape(U)[0]

            # batch indices
            loop_until = tf.range(0, batch_size, dtype=tf.int32)

            # initial estimated positions
            initial_guesses = tf.zeros([2, batch_size], dtype=tf.int32)
            s, e = tf.split(initial_guesses, 2, axis=0)

            print('s:', s.get_shape())

            # Get U vectors of starting indexes
            fn = lambda idx: select(U, s, idx)
            u_s = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)
            u_s = tf.reshape(u_s, shape=[batch_size, 2 * self.hidden_size])

            # Get U vectors of ending indexes
            fn = lambda idx: select(U, e, idx)
            u_e = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)
            u_e = tf.reshape(u_e, shape=[batch_size, 2 * self.hidden_size])

        print('u_s:', u_s.get_shape())
        print('u_e:', u_e.get_shape())

        with tf.variable_scope('highway_init'):
            highway_alpha = highway_maxout(self.hidden_size, self.pool_size)
            highway_beta = highway_maxout(self.hidden_size, self.pool_size)

        with tf.variable_scope('decoder'):
            # LSTM for decoding
            lstm_dec = LSTMCell(self.hidden_size)

            for step in range(self.max_decode_steps):
                if step > 0:
                    tf.reuse_variables()
                # single step lstm
                _input = tf.concat([u_s, u_e], axis=1)
                print('_input:', _input.get_shape())

                _, h = tf.contrib.rnn.static_rnn(lstm_dec, [_input], dtype=tf.float32)
               
                h_state = tf.concat(h, axis=1)

                print('h_state:', h_state.get_shape())
                with tf.variable_scope('highway_alpha'):
                    # compute start position first
                    fn = lambda u_t: highway_alpha(u_t, h_state, u_s, u_e)
                    alpha = tf.map_fn(lambda u_t: fn(u_t), U, dtype=tf.float32)
                    s = tf.reshape(tf.argmax(alpha, axis=0), [batch_size])
                    
                    # update start guess
                    fn = lambda idx: select(U, s, idx)
                    u_s = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)

                with tf.variable_scope('highway_beta'):
                    # compute end position next
                    fn = lambda u_t: highway_beta(u_t, h_state, u_s, u_e)
                    beta = tf.map_fn(lambda u_t: fn(u_t), U, dtype=tf.float32)
                    e = tf.reshape(tf.argmax(beta, axis=0), [batch_size])
                    
                    # update end guess
                    fn = lambda idx: select(U, e, idx)
                    u_e = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)

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

    