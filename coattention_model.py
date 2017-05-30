import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import GRUCell, LSTMCell
import numpy as np

# Attempt at recreating: https://arxiv.org/pdf/1611.01604.pdf
class Model:
    def __init__(self, config, max_x, max_q):
        self.model_name = 'coattention'
        self.emb_size = config.emb_size
        self.hidden_size = config.hidden_size
        self.max_x = max_x
        self.max_q = max_q
        self.saver = None

    def build(self, x, x_len, q, q_len, y_begin, y_end, emb_mat, keep_prob):        
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

            Q, _ = tf.nn.dynamic_rnn(lstm_enc, question, sequence_length=q_len, dtype=tf.float32) # (batch_size, max_q, hidden_size)
            Q = tf.map_fn(lambda x: fn(x), Q, dtype=tf.float32)
            Q = tf.transpose(Q, perm=[0, 2, 1]) # (batch_size, hidden_size, max_q)
            tf.summary.histogram('Q', Q)   

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

        alpha = tf.layers.dense(U, 1, name='alpha')
        alpha = tf.reshape(alpha, [-1, self.max_x + 1])
        beta = tf.layers.dense(U, 1, name='beta')
        beta = tf.reshape(beta, [-1, self.max_x + 1])

        with tf.variable_scope('loss'):
            loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_begin, logits=alpha), name='beginning_loss')
            loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_end, logits=beta), name='ending_loss')
            loss = loss1 + loss2
        with tf.variable_scope('accuracy'):
            acc1 = tf.reduce_mean(tf.cast(tf.equal(y_begin, tf.cast(tf.argmax(alpha, 1), 'int32')), 'float'), name='beginning_accuracy')
            acc2 = tf.reduce_mean(tf.cast(tf.equal(y_end, tf.cast(tf.argmax(beta, 1), 'int32')), 'float'), name='ending_accuracy')

        self.loss = loss
        # self.loss = self._loss_multitask(self._alpha, y_begin, self._beta, y_end)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('loss1', loss1)
        tf.summary.scalar('loss2', loss2)
        tf.summary.scalar('accuracy1', acc1)
        tf.summary.scalar('accuracy2', acc2)
        
        self.logits1 = alpha
        self.logits2 = beta
        
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
