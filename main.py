import argparse
import tensorflow as tf


from model import Model
from data import Data

def get_parser():
    parser = argparse.ArgumentParser()
    home = "" #os.path.expanduser("~")
    parser.add_argument('--keep_prob', default=0.8)
    parser.add_argument('--hidden_size', default=100)
    parser.add_argument('--dim_size', default=50) # this could be 50 (171.4 MB), 100 (347.1 MB), 200 (693.4 MB), or 300 (1 GB)
    parser.add_argument('--train_path', default='./datasets/msmarco/train/location.json')
    parser.add_argument('--val_path', default='./datasets/msmarco/dev/location.json')
    parser.add_argument('--reference_path', default='./eval/references.json')
    parser.add_argument('--candidate_path', default='./eval/candidates.json')
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--learning_rate', default=0.5)

    return parser

def main():
    parser = get_parser()
    config = parser.parse_args()

    data = Data(config)

    model = Model(config)

    tX = data.tX
    tXq = data.tXq
    embeddings = data.embeddings

    # shape = batch_size by num_features
    x = tf.placeholder(tf.int32, shape=[data.batch_size, tX[0].shape[0]], name='x')
    x_len = tf.placeholder(tf.int32, shape=[None], name='x_len')
    q = tf.placeholder(tf.int32, shape=[data.batch_size, tXq[0].shape[0]], name='q')
    q_len = tf.placeholder(tf.int32, shape=[None], name='q_len')

    outputs = model.build(x, x_len, q, q_len, embeddings)

    # Place holder for just index of answer within context  
    y_begin = tf.placeholder(tf.int32, [None], name='y_begin')
    y_end = tf.placeholder(tf.int32, [None], name='y_end')

    logits1, logits2 = outputs['logits_start'], outputs['logits_end']
    loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_begin, logits=logits1))
    loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_end, logits=logits2))
    loss = loss1 + loss2
    acc1 = tf.reduce_mean(tf.cast(tf.equal(y_begin, tf.cast(tf.argmax(logits1, 1), 'int32')), 'float'))
    acc2 = tf.reduce_mean(tf.cast(tf.equal(y_end, tf.cast(tf.argmax(logits2, 1), 'int32')), 'float'))

    train_step = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(loss)

    number_of_batches = data.get_num_batches()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1):#config.epochs * number_of_batches):
            batch = data.get_train_batch()

            train_step.run(feed_dict={x: batch['tX'],
                                    x_len: [config.max_context_size] * data.batch_size,
                                    q: batch['tXq'],
                                    q_len: [config.max_ques_size] * data.batch_size,
                                    y_begin: batch['tYBegin'],
                                    y_end: batch['tYEnd']})

            if i % 20 == 0:
                train_accuracy_begin = acc1.eval(feed_dict={x: batch['tX'],
                                        x_len: [config.max_context_size] * data.batch_size,
                                        q: batch['tXq'],
                                        q_len: [config.max_ques_size] * data.batch_size,
                                        y_begin: batch['tYBegin'],
                                        y_end: batch['tYEnd']})
                train_accuracy_end = acc2.eval(feed_dict={x: batch['tX'],
                                        x_len: [config.max_context_size] * data.batch_size,
                                        q: batch['tXq'],
                                        q_len: [config.max_ques_size] * data.batch_size,
                                        y_begin: batch['tYBegin'],
                                        y_end: batch['tYEnd']})

                print('step %d, beginning accuracy %g end accuracy %g' % (i, train_accuracy_begin, train_accuracy_end))


        # Print out answers for one of the batches
        batch = data.get_val_batch()

        prediction_begin = tf.cast(tf.argmax(logits1, 1), 'int32')
        prediction_end = tf.cast(tf.argmax(logits2, 1), 'int32')


        begin = prediction_begin.eval(feed_dict={x: batch['vX'],
                                                x_len: [config.max_context_size] * data.batch_size,
                                                q: batch['vXq'],
                                                q_len: [config.max_ques_size] * data.batch_size,
                                                y_begin: batch['vYBegin'],
                                                y_end: batch['vYEnd']})

        end = prediction_end.eval(feed_dict={x: batch['vX'],
                                             x_len: [config.max_context_size] * data.batch_size,
                                             q: batch['vXq'],
                                             q_len: [config.max_ques_size] * data.batch_size,
                                             y_begin: batch['vYBegin'],
                                             y_end: batch['vYEnd']})
        for j in range(len(begin)):
            print(batch['vQuestion'][j])
            print(batch['vContext'][j][begin[j] : end[j] + 1])
            print()

if __name__ == "__main__":
    main()