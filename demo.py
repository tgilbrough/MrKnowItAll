
import tensorflow as tf
import nltk
import numpy as np

import argparse

import main
from data import Data


def tokenize(string):
    string = string.replace("''", '" ')
    string = string.replace("``", '" ')
    return [token.replace("``", '"').replace("''", '"')
            for token in nltk.word_tokenize(string.lower())]


def pad(vector, length):
    if len(vector) < length:
        return vector + [0] * (length - len(vector))
    return vector


def _main():
    parser = main.get_parser()
    parser.add_argument('model_path', help='directory of saved model')
    config = parser.parse_args()
    main.fill_paths(config)
    model_path = ('{}{}model.meta'
                  .format(config.model_path,
                          '' if config.model_path[-1] == '/' else '/'))
    # data = Data(config)

    with tf.Session() as sess:
        print('Loading model from {}'.format(model_path))
        saver = tf.train.import_meta_graph(model_path)
        saver.restore(sess, tf.train.latest_checkpoint(config.model_path))

        vocab = tf.get_collection('vocab')
        word_index = {c.decode('utf-8'): i + 1 for i, c in enumerate(vocab)}
        dimensions = tf.get_collection('dimensions')
        max_context_size = dimensions[0]
        max_question_size = dimensions[1]

        def vectorize(passage, length):
            'Discards unknown words.'
            return pad([word_index[token]
                        for token in tokenize(passage)
                        if token in word_index],
                       length)

        context_string = '''1. Get help from a doctor now ›. under ribs: The lungs in the front and back are inside the rib cage. This is why doctors will place a stethoscope on the back as well as the front to evaluate the function of the lungs. ...Read more. 1 Where are your kidneys located on your back in women. 2  Where are the lungs located in the back. 3  Where are lungs located in your body. 4  Ask a doctor a question free online. 5  Where are the lungs located in the human body.'''
        question_string = 'where are the lungs located in the back'
        context = vectorize(context_string, max_context_size)
        question = vectorize(question_string, max_question_size)

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('x:0')
        x_len = graph.get_tensor_by_name('x_len:0')
        q = graph.get_tensor_by_name('q:0')
        q_len = graph.get_tensor_by_name('q_len:0')
        y_begin = graph.get_tensor_by_name('y_begin:0')
        y_end = graph.get_tensor_by_name('y_end:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')

        feed_dict = {
            x: np.array([context], dtype=np.int32),
            x_len: [len(context)],
            q: np.array([question], dtype=np.int32),
            q_len: [len(question)],
            y_begin: [1],
            y_end: [2],
            keep_prob: 1.0
        }

        logits = tf.get_collection('logits')
        prediction_begin = tf.cast(tf.argmax(logits[0], 1), 'int32')
        prediction_end = tf.cast(tf.argmax(logits[1], 1), 'int32')

        begin, end = sess.run([prediction_begin, prediction_end],
                              feed_dict=feed_dict)

        print(begin)
        print(end)
        print(context_string[begin[0]:end[0]])


if __name__ == '__main__':
    _main()
