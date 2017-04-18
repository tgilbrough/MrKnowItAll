import argparse
import tensorflow as tf

from model import Model
from data import Data

def get_parser():
    parser = argparse.ArgumentParser()
    home = "" #os.path.expanduser("~")
    parser.add_argument('--keep_prob', type=float, default=0.8)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--emb_size', type=int, default=50) # this could be 50 (171.4 MB), 100 (347.1 MB), 200 (693.4 MB), or 300 (1 GB)
    parser.add_argument('--train_path', default='./datasets/msmarco/train/location.json')
    parser.add_argument('--val_path', default='./datasets/msmarco/dev/location.json')
    parser.add_argument('--reference_path', default='./eval/references.json')
    parser.add_argument('--candidate_path', default='./eval/candidates.json')
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--learning_rate', '-l', type=float, default=0.5)

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
    x = tf.placeholder(tf.int32, shape=[None, tX[0].shape[0]], name='x')
    x_len = tf.placeholder(tf.int32, shape=[None], name='x_len')
    q = tf.placeholder(tf.int32, shape=[None, tXq[0].shape[0]], name='q')
    q_len = tf.placeholder(tf.int32, shape=[None], name='q_len')
    keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

    outputs = model.build(x, x_len, q, q_len, embeddings, keep_prob)

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

    number_of_train_batches = data.getNumTrainBatches()
    number_of_val_batches = data.getNumValBatches()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(config.epochs):
            print('Epoch {}/{}'.format(e + 1, config.epochs))
            for i in range(number_of_train_batches):
                batch = data.getTrainBatch()

                train_step.run(feed_dict={x: batch['tX'],
                                        x_len: [config.max_context_size] * len(batch['tX']),
                                        q: batch['tXq'],
                                        q_len: [config.max_ques_size] * len(batch['tX']),
                                        y_begin: batch['tYBegin'],
                                        y_end: batch['tYEnd'], 
                                        keep_prob: config.keep_prob})

                if i % 20 == 0:
                    acc_begin, acc_end = sess.run([acc1, acc2], feed_dict={x: batch['tX'],
                                            x_len: [config.max_context_size] * len(batch['tX']),
                                            q: batch['tXq'],
                                            q_len: [config.max_ques_size] * len(batch['tX']),
                                            y_begin: batch['tYBegin'],
                                            y_end: batch['tYEnd'],
                                            keep_prob: 1.0})

                    print('step {}, beginning accuracy {} end accuracy {}'.format(i, acc_begin, acc_end))


        # Print out answers for one of the batches
        vContext = []
        vQuestionID = []
        predictedBegin = []
        predictedEnd = []
        trueBegin = []
        trueEnd = []
        for i in range(number_of_val_batches):
            batch = data.getValBatch()

            prediction_begin = tf.cast(tf.argmax(logits1, 1), 'int32')
            prediction_end = tf.cast(tf.argmax(logits2, 1), 'int32')

            begin, end = sess.run([prediction_begin, prediction_end], feed_dict={x: batch['vX'],
                                                    x_len: [config.max_context_size] * len(batch['vX']),
                                                    q: batch['vXq'],
                                                    q_len: [config.max_ques_size] * len(batch['vX']),
                                                    y_begin: batch['vYBegin'],
                                                    y_end: batch['vYEnd'], 
                                                    keep_prob: 1.0})

            for j in range(len(begin)):
                vContext.append(batch['vContext'][j])
                vQuestionID.append(batch['vQuestionID'][j])
                predictedBegin.append(begin[j])
                predictedEnd.append(end[j])
                trueBegin.append(batch['vYBegin'][j])
                trueEnd.append(batch['vYEnd'][j])
                #print(batch['vQuestion'][j])
                #print(batch['vContext'][j][begin[j] : end[j] + 1])
                #print()
            
        data.saveAnswersForEval(config.reference_path, config.candidate_path, vContext, vQuestionID, predictedBegin, predictedEnd, trueBegin, trueEnd)

if __name__ == "__main__":
    main()