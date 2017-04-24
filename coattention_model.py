import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import GRUCell, LSTMCell

# Attempt at recreating: https://arxiv.org/pdf/1611.01604.pdf
class Model:
    def __init__(self, config, max_x, max_q):
        self.model_name = 'coattention'
        self.emb_size = config.emb_size
        self.hidden_size = config.hidden_size
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

        with tf.variable_scope('encoding_context'):
            D, _ = tf.nn.dynamic_rnn(lstm_enc, context, sequence_length=x_len, dtype=tf.float32)
            D = tf.transpose(D, perm=[0, 2, 1]) 
            tf.summary.histogram("D", D)

        with tf.variable_scope('encoding_question'):
            q, _ = tf.nn.dynamic_rnn(lstm_enc, question, sequence_length=q_len, dtype=tf.float32)
            q = tf.transpose(q, perm=[0, 2, 1]) 
            tf.summary.histogram("Q'", q)

            # map to separate question space
            q_weights = tf.get_variable(name='q_weights', shape=[self.hidden_size, self.hidden_size], dtype=tf.float32)
            res = tf.map_fn(lambda x: tf.matmul(q_weights, x), q)
            q_biases = tf.get_variable(name='q_bias', shape=[self.hidden_size, self.max_q], dtype=tf.float32)
            
            Q = tf.tanh(tf.map_fn(lambda x: tf.add(x, q_biases), res))    
            tf.summary.histogram("Q", Q)   
            

        print('D:', D.get_shape())
        print('Q:', Q.get_shape())
        

        with tf.variable_scope('affinity_mat'):
            L = tf.matmul(tf.transpose(D, perm=[0, 2, 1]), Q, name='L')

        with tf.variable_scope('normalize_aff'):
            Aq = tf.nn.softmax(L, name='Aq')
            Ad = tf.nn.softmax(tf.transpose(L), name='Ad')

        with tf.variable_scope('attention_contexts'):
            Cq = tf.matmul(D, Aq, name='Cq')

        with tf.variable_scope('attention_questions'):
            Cd = tf.concat([Q, Cq], axis=1)
            Cd = tf.matmul(Cd, Ad, name='Cd')

        co_att = tf.concat([D, Cd], axis=2)
        
        with tf.variable_scope('encoder'):
            lstm_fw_cell = LSTMCell(self.hidden_size)
            lstm_bw_cell = LSTMCell(self.hidden_size)

            u, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs=co_att, sequence_length=x_len, dtype=tf.float32)
            
            u_fw, u_bw = u
            u = tf.concat([u_fw, u_bw], axis=2)

        with tf.variable_scope('decoder'):
            lstm_dec = LSTMCell(self.hidden_size)
            question_fw, question_bw = outputs_question
            question_output = tf.concat([question_fw, question_bw], axis=2)
        



        

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
