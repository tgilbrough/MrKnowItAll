import argparse
import tensorflow as tf
from tqdm import tqdm
import os

import baseline_model
import attention_model

from data import Data

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep_prob', '-kp', type=float, default=0.7)
    parser.add_argument('--hidden_size', '-hs', type=int, default=100)
    parser.add_argument('--emb_size', '-es', type=int, default=50) # this could be 50 (171.4 MB), 100 (347.1 MB), 200 (693.4 MB), or 300 (1 GB)
    parser.add_argument('--question_type', '-q', default='location',
                        choices=['description', 'entity', 'location', 'numeric', 'person'])
    parser.add_argument('--train_path', default='./datasets/msmarco/train/')
    parser.add_argument('--val_path', default='./datasets/msmarco/dev/')
    parser.add_argument('--reference_path', default='./eval/references.json')
    parser.add_argument('--candidate_path', default='./eval/candidates.json')
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01)
    parser.add_argument('--num_threads', '-t', type=int, default=4)
    parser.add_argument('--model_save_dir', default='./saved_models')
    parser.add_argument('--load_model', '-l', type=int, default=0)
    parser.add_argument('--model', '-m', default='baseline')
    parser.add_argument('--tensorboard_name', '-tn', default=None)

    return parser

def main():
    parser = get_parser()
    config = parser.parse_args()
    config.train_path = '{}{}.json'.format(config.train_path, config.question_type)
    config.val_path = '{}{}.json'.format(config.val_path, config.question_type)

    load_model = config.load_model

    tf.reset_default_graph()

    data = Data(config)

    if config.model == 'baseline':
        model = baseline_model.Model(config, data.max_context_size, data.max_ques_size)
        print("Using baseline model")
    elif config.model == 'attention':
        model = attention_model.Model(config, data.max_context_size, data.max_ques_size)
        print("Using attention model")
    else:
        raise ValueError('Specified invalid model')

    if config.tensorboard_name is None:
        config.tensorboard_name = model.model_name
    tensorboard_path = './tensorboard_models/' + config.tensorboard_name
    save_model_path = config.model_save_dir + '/' + config.tensorboard_name
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    print('Building tensorflow computation graph...')

    # shape = batch_size by num_features
    x = tf.placeholder(tf.int32, shape=[None, data.tX[0].shape[0]], name='x')
    x_len = tf.placeholder(tf.int32, shape=[None], name='x_len')
    q = tf.placeholder(tf.int32, shape=[None, data.tXq[0].shape[0]], name='q')
    q_len = tf.placeholder(tf.int32, shape=[None], name='q_len')
    keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

    outputs = model.build(x, x_len, q, q_len, data.embeddings, keep_prob)

    print('Computation graph completed.')

    # Place holder for just index of answer within context
    y_begin = tf.placeholder(tf.int32, [None], name='y_begin')
    y_end = tf.placeholder(tf.int32, [None], name='y_end')

    with tf.variable_scope('loss'):
        logits1, logits2 = outputs['logits_start'], outputs['logits_end']
        loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_begin, logits=logits1), name='beginning_loss')
        loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_end, logits=logits2), name='ending_loss')
        loss = loss1 + loss2
    with tf.variable_scope('accuracy'):
        acc1 = tf.reduce_mean(tf.cast(tf.equal(y_begin, tf.cast(tf.argmax(logits1, 1), 'int32')), 'float'), name='beginning_accuracy')
        acc2 = tf.reduce_mean(tf.cast(tf.equal(y_end, tf.cast(tf.argmax(logits2, 1), 'int32')), 'float'), name='ending_accuracy')

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("loss1", loss1)
    tf.summary.scalar("loss2", loss2)
    tf.summary.scalar("accuracy1", acc1)
    tf.summary.scalar("accuracy2", acc2)
    merged_summary = tf.summary.merge_all()

    train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(loss)

    number_of_train_batches = data.getNumTrainBatches()
    number_of_val_batches = data.getNumValBatches()

    # For tensorboard
    train_writer = tf.summary.FileWriter(tensorboard_path + '/train')
    val_writer = tf.summary.FileWriter(tensorboard_path + '/dev')

    # For saving models
    saver = tf.train.Saver()
    min_val_loss = float('Inf')

    with tf.Session() as sess:
        train_writer.add_graph(sess.graph)
        val_writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())

        for e in range(config.epochs):
            print('Epoch {}/{}'.format(e + 1, config.epochs))
            for i in tqdm(range(number_of_train_batches)):
                trainBatch = data.getRandomTrainBatch()

                feed_dict={x: trainBatch['tX'],
                            x_len: [data.max_context_size] * len(trainBatch['tX']),
                            q: trainBatch['tXq'],
                            q_len: [data.max_ques_size] * len(trainBatch['tX']),
                            y_begin: trainBatch['tYBegin'],
                            y_end: trainBatch['tYEnd'],
                            keep_prob: config.keep_prob}
                sess.run(train_step, feed_dict=feed_dict)


            # Record results for tensorboard, once per epoch
            feed_dict={x: trainBatch['tX'],
                    x_len: [data.max_context_size] * len(trainBatch['tX']),
                    q: trainBatch['tXq'],
                    q_len: [data.max_ques_size] * len(trainBatch['tX']),
                    y_begin: trainBatch['tYBegin'],
                    y_end: trainBatch['tYEnd'],
                    keep_prob: 1.0}
            train_sum = sess.run(merged_summary, feed_dict=feed_dict)

            valBatch = data.getRandomValBatch()
            feed_dict={x: valBatch['vX'],
                    x_len: [data.max_context_size] * len(valBatch['vX']),
                    q: valBatch['vXq'],
                    q_len: [data.max_ques_size] * len(valBatch['vX']),
                    y_begin: valBatch['vYBegin'],
                    y_end: valBatch['vYEnd'],
                    keep_prob: 1.0}
            val_sum, val_loss  = sess.run([merged_summary, loss], feed_dict=feed_dict)
            if val_loss < min_val_loss:
                saver.save(sess, save_model_path + '/model')
                min_val_loss = val_loss

            train_writer.add_summary(train_sum, e)
            val_writer.add_summary(val_sum, e)

        # Load best graph on validation data
        new_saver = tf.train.import_meta_graph(save_model_path + '/model.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(save_model_path))

        # Print out answers for one of the batches
        vContext = []
        vQuestionID = []
        predictedBegin = []
        predictedEnd = []
        trueBegin = []
        trueEnd = []

        begin_corr = 0
        end_corr = 0
        total = 0

        for i in range(number_of_val_batches):
            valBatch = data.getValBatch()

            prediction_begin = tf.cast(tf.argmax(logits1, 1), 'int32')
            prediction_end = tf.cast(tf.argmax(logits2, 1), 'int32')


            feed_dict={x: valBatch['vX'],
                            x_len: [data.max_context_size] * len(valBatch['vX']),
                            q: valBatch['vXq'],
                            q_len: [data.max_ques_size] * len(valBatch['vX']),
                            y_begin: valBatch['vYBegin'],
                            y_end: valBatch['vYEnd'],
                            keep_prob: 1.0}
            begin, end = sess.run([prediction_begin, prediction_end], feed_dict=feed_dict)


            for j in range(len(begin)):
                vContext.append(valBatch['vContext'][j])
                vQuestionID.append(valBatch['vQuestionID'][j])
                predictedBegin.append(begin[j])
                predictedEnd.append(end[j])
                trueBegin.append(valBatch['vYBegin'][j])
                trueEnd.append(valBatch['vYEnd'][j])

                begin_corr += int(begin[j] == valBatch['vYBegin'][j])
                end_corr += int(end[j] == valBatch['vYEnd'][j])
                total += 1

                #print(batch['vQuestion'][j])
                #print(batch['vContext'][j][begin[j] : end[j] + 1])
                #print()

        print('Validation Data:')
        print('begin accuracy: {}'.format(float(begin_corr) / total))
        print('end accuracy: {}'.format(float(end_corr) / total))

        data.saveAnswersForEval(config.reference_path, config.candidate_path, vContext, vQuestionID, predictedBegin, predictedEnd, trueBegin, trueEnd)

if __name__ == "__main__":
    main()
