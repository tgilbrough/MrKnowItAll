import argparse
import tensorflow as tf
from tqdm import tqdm
import os
import sys
import json
import numpy as np

import bidaf_model_multi

from data_multi import Data

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t', type=int, default=1)
    parser.add_argument('--keep_prob', '-kp', type=float, default=0.5)
    parser.add_argument('--hidden_size', '-hs', type=int, default=50)
    parser.add_argument('--emb_size', '-es', type=int, default=50) # this could be 50 (171.4 MB), 100 (347.1 MB), 200 (693.4 MB), or 300 (1 GB)
    parser.add_argument('--question_type', '-q', default='location',
                        choices=['description', 'entity', 'location', 'numeric', 'person'])
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01)
    parser.add_argument('--load_model', '-l', type=int, default=0)
    parser.add_argument('--model', '-m', default='bidaf', choices=['bidaf'])
    parser.add_argument('--tensorboard_name', '-tn', default=None)
    parser.add_argument('--cell', '-c', default='lstm')
    parser.add_argument('--highway_network', '-hwn', type=int, default=1)

    return parser

def main():
    parser = get_parser()
    config = parser.parse_args()
    config.train_path = '{}{}.json'.format('./datasets/msmarco/train/', config.question_type)
    config.val_path = '{}{}.json'.format('./datasets/msmarco/dev/', config.question_type)

    load_model = config.load_model

    tf.reset_default_graph()

    data = Data(config)
    
    model = bidaf_model_multi.Model(config, data.max_context_size, data.max_ques_size)

    if config.tensorboard_name is None:
        config.tensorboard_name = model.model_name
    tensorboard_path = './tensorboard_models/' + config.tensorboard_name
    save_model_path = './saved_models/' + config.tensorboard_name
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    print('Building tensorflow computation graph...')

    # shape = batch_size by num_features
    x = [tf.placeholder(tf.int32, shape=[None, data.max_context_size], name=('x'+str(i))) for i in range(data.max_passages)]
    x_weights = tf.placeholder(tf.float32, shape=[None, data.max_passages], name='x_weights')
    x_len = tf.placeholder(tf.int32, shape=[None], name='x_len')
    q = tf.placeholder(tf.int32, shape=[None, data.tXq[0].shape[0]], name='q')
    q_len = tf.placeholder(tf.int32, shape=[None], name='q_len')
    keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

    # Place holder for just index of answer within context
    y_begin = tf.placeholder(tf.int32, [None], name='y_begin')
    y_end = tf.placeholder(tf.int32, [None], name='y_end')

    model.build(x, x_weights, x_len, q, q_len, y_begin, y_end, data.embeddings, keep_prob)

    print('Computation graph completed.')

    train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(model.loss)

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

        if config.train:
            for e in range(config.epochs):
                print('Epoch {}/{}'.format(e + 1, config.epochs))
                for i in tqdm(range(number_of_train_batches)):
                    trainBatch = data.getRandomTrainBatch()

                    feed_dict={
                                x_weights: trainBatch['tXPassWeights'],
                                x_len: [data.max_context_size for i in range(len(trainBatch['tX']))],
                                q: trainBatch['tXq'],
                                q_len: [data.max_ques_size for i in range(len(trainBatch['tXq']))],
                                y_begin: trainBatch['tYBegin'],
                                y_end: trainBatch['tYEnd'],
                                keep_prob: config.keep_prob}
                    tX_trans = np.transpose(trainBatch['tX'], (1, 0, 2))
                    for j in range(len(x)):
                        feed_dict[x[j]] = tX_trans[j]
                    sess.run(train_step, feed_dict=feed_dict)

                # Record results for tensorboard, once per epoch
                feed_dict={
                        x_weights: trainBatch['tXPassWeights'],
                        x_len: [len(trainBatch['tX'][i]) for i in range(len(trainBatch['tX']))],
                        q: trainBatch['tXq'],
                        q_len: [len(trainBatch['tXq'][i]) for i in range(len(trainBatch['tXq']))],
                        y_begin: trainBatch['tYBegin'],
                        y_end: trainBatch['tYEnd'],
                        keep_prob: 1.0}
                tX_trans = np.transpose(trainBatch['tX'], (1, 0, 2))
                for j in range(len(x)):
                    feed_dict[x[j]] = tX_trans[j]
                train_sum = sess.run(model.merged_summary, feed_dict=feed_dict)

                valBatch = data.getRandomValBatch()
                feed_dict={
                        x_weights: valBatch['vXPassWeights'],
                        x_len: [len(valBatch['vX'][i]) for i in range(len(valBatch['vX']))],
                        q: valBatch['vXq'],
                        q_len: [len(valBatch['vXq'][i]) for i in range(len(valBatch['vXq']))],
                        y_begin: valBatch['vYBegin'],
                        y_end: valBatch['vYEnd'],
                        keep_prob: 1.0}
                vX_trans = np.transpose(valBatch['vX'], (1, 0, 2))
                for j in range(len(x)):
                    feed_dict[x[j]] = vX_trans[j]
                val_sum, val_loss  = sess.run([model.merged_summary, model.loss], feed_dict=feed_dict)
                if val_loss < min_val_loss:
                    saver.save(sess, save_model_path + '/model')
                    min_val_loss = val_loss

                train_writer.add_summary(train_sum, e)
                val_writer.add_summary(val_sum, e)

        # Load best graph on validation data
        try:
            new_saver = tf.train.import_meta_graph(save_model_path + '/model.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint(save_model_path))
        except:
            print('Must train model first with --train flag')
            sys.exit()

        # Print out answers for one of the batches
        vContext = []
        vQuestionID = []
        predictedBegin = []
        trueBegin = []
        
        begin_corr = 0
        total = 0

        f = open('multi_start.json', 'w', encoding='utf-8')
        

        for i in range(number_of_val_batches):
            valBatch = data.getValBatch()

            prediction_begin = tf.cast(tf.argmax(model.logits1, 1), 'int32')

            feed_dict={
                        x_weights: valBatch['vXPassWeights'],
                        x_len: [len(valBatch['vX'][i]) for i in range(len(valBatch['vX']))],
                        q: valBatch['vXq'],
                        q_len: [len(valBatch['vXq'][i]) for i in range(len(valBatch['vXq']))],
                        y_begin: valBatch['vYBegin'],
                        y_end: valBatch['vYEnd'],
                        keep_prob: 1.0}
            vX_trans = np.transpose(valBatch['vX'], (1, 0, 2))
            for j in range(len(x)):
                feed_dict[x[j]] = vX_trans[j]
            (begin,) = sess.run([prediction_begin], feed_dict=feed_dict)


            for j in range(len(begin)):
                sample = {}
                sample['query_id'] = valBatch['vQuestionID'][j]
                sample['start'] = np.asscalar(begin[j])

                print(json.dumps(sample, ensure_ascii=False), file=f)

                begin_corr += int(begin[j] == valBatch['vYBegin'][j])
                total += 1


        print('Validation Data:')
        print('begin accuracy: {}'.format(float(begin_corr) / total))

        f.close()

if __name__ == "__main__":
    main()
