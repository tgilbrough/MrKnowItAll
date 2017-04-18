
import tensorflow as tf

import json
from math import sqrt

from glove import Glove
glove = Glove()


def load_documents(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def load_passages(path):
    print('Loading passages from {}...'.format(path))

    passages = []
    selected = []

    for document in load_documents(path):
        query_glove = glove.paragraph(document['query'])
        for passage in document['passages']:
            passage_glove = glove.paragraph(passage['passage_text'])
            passages.append(query_glove + passage_glove)
            selected.append([passage['is_selected'], 1 - passage['is_selected']])

    print('  Loaded {} passages.'.format(len(passages)))

    return passages, selected


train_x, train_y = load_passages('datasets/msmarco/dev_v1.1.json')
# test_x, test_y = load_passages('datasets/msmarco/test_public_v1.1.json')

n_features = glove.dim * 2
n_labels = 2
n_epochs = 100

learning_rate = tf.train.exponential_decay(learning_rate=0.0008,
                                           global_step=1,
                                           decay_steps=len(train_x),
                                           decay_rate=0.95,
                                           staircase=True)

x = tf.placeholder(tf.float32, [None, n_features])
y_gold = tf.placeholder(tf.float32, [None, n_labels])

stddev = (sqrt(6 / (n_features + n_labels + 1)))
weights = tf.Variable(tf.random_normal([n_features, n_labels],
                                       mean=0,
                                       stddev=stddev,
                                       name='weights'))

bias = tf.Variable(tf.random_normal([1, n_labels],
                                    mean=0,
                                    stddev=stddev,
                                    name='bias'))

init_op = tf.global_variables_initializer()

apply_weights_op = tf.matmul(x, weights, name='apply_weights')
add_bias_op = tf.add(apply_weights_op, bias, name='add_bias')
activation_op = tf.nn.sigmoid(add_bias_op, name='activation')
cost_op = tf.nn.l2_loss(activation_op - y_gold, name='squared_error_cost')
training_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_op)

n_correct_op = tf.equal(tf.argmax(activation_op, 1), tf.argmax(y_gold, 1))
accuracy_op = tf.reduce_mean(tf.cast(n_correct_op, 'float'))

with tf.Session() as session:
    session.run(init_op)

    for i in range(n_epochs):
        feed_dict = {x: train_x, y_gold: train_y}
        step = session.run(training_op, feed_dict=feed_dict)

        if i % 10 == 0:
            print(i)
            accuracy, cost = session.run([accuracy_op, cost_op], feed_dict=feed_dict)
            print(accuracy)
            print(cost)
