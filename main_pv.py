import argparse
import tensorflow as tf
from tqdm import tqdm
import os

import baseline_model_pv

from data_pv import Data

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep_prob', '-kp', type=float, default=0.5)
    parser.add_argument('--hidden_size', '-hs', type=int, default=50)
    parser.add_argument('--emb_size', '-es', type=int, default=50) # this could be 50 (171.4 MB), 100 (347.1 MB), 200 (693.4 MB), or 300 (1 GB)
    parser.add_argument('--question_type', '-q', default='location',
                        choices=['description', 'entity', 'location', 'numeric', 'person'])
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01)
    parser.add_argument('--load_model', '-l', type=int, default=0)
    parser.add_argument('--model', '-m', default='baseline', 
                        choices=['baseline'])
    parser.add_argument('--tensorboard_name', '-tn', default=None)

    return parser

def main():
    parser = get_parser()
    config = parser.parse_args()
    config.train_path = '{}{}.json'.format('./datasets/msmarco/train/', config.question_type)
    config.val_path = '{}{}.json'.format('./datasets/msmarco/dev/', config.question_type)

    load_model = config.load_model

    tf.reset_default_graph()

    data = Data(config)

    if config.model == 'baseline':
        model = baseline_model_pv.Model(config, data.max_context_size, data.max_ques_size)
        print("Using baseline model")
    else:
        raise Exception('Please provide which model to use with the -q argument')

    if config.tensorboard_name is None:
        config.tensorboard_name = model.model_name
    tensorboard_path = './tensorboard_models/' + config.tensorboard_name + '_pv'
    save_model_path = './saved_models/' + config.tensorboard_name + '_pv'
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    print('Building tensorflow computation graph...')

    # shape = batch_size by num_features
    x = tf.placeholder(tf.int32, shape=[None, data.tX[0].shape[0]], name='x')
    x_len = tf.placeholder(tf.int32, shape=[None], name='x_len')
    q = tf.placeholder(tf.int32, shape=[None, data.tXq[0].shape[0]], name='q')
    q_len = tf.placeholder(tf.int32, shape=[None], name='q_len')
    keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

    # Place holder for just index of answer within context
    y = tf.placeholder(tf.int32, [None], name='y')

    model.build(x, x_len, q, q_len, y, data.embeddings, keep_prob)

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

        for e in range(config.epochs):
            print('Epoch {}/{}'.format(e + 1, config.epochs))
            for i in tqdm(range(number_of_train_batches)):
                trainBatch = data.getRandomTrainBatch()

                feed_dict={x: trainBatch['tX'],
                    x_len: [len(trainBatch['tX'][i]) for i in range(len(trainBatch['tX']))],
                            q: trainBatch['tXq'],
                            q_len: [len(trainBatch['tXq'][i]) for i in range(len(trainBatch['tXq']))],
                    y: trainBatch['tY'],
                    keep_prob: config.keep_prob}

                sess.run(train_step, feed_dict=feed_dict)

            # Record results for tensorboard, once per epoch
            feed_dict={x: trainBatch['tX'],
                    x_len: [len(trainBatch['tX'][i]) for i in range(len(trainBatch['tX']))],
                    q: trainBatch['tXq'],
                    q_len: [len(trainBatch['tXq'][i]) for i in range(len(trainBatch['tXq']))],
                    y: trainBatch['tY'],
                    keep_prob: 1.0}
            train_sum = sess.run(model.merged_summary, feed_dict=feed_dict)

            valBatch = data.getRandomValBatch()
            feed_dict={x: valBatch['vX'],
                    x_len: [len(valBatch['vX'][i]) for i in range(len(valBatch['vX']))],
                    q: valBatch['vXq'],
                    q_len: [len(valBatch['vXq'][i]) for i in range(len(valBatch['vX']))],
                    y: valBatch['vY'],
                    keep_prob: 1.0}
            val_sum, val_loss  = sess.run([model.merged_summary, model.loss], feed_dict=feed_dict)
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

        total = 0

        for i in range(number_of_val_batches):
            valBatch = data.getValBatch()

            prediction = tf.cast(tf.argmax(model.logits, 1), 'int32')

            feed_dict={x: valBatch['vX'],
                    x_len: [len(valBatch['vX'][i]) for i in range(len(valBatch['vX']))],
                    q: valBatch['vXq'],
                    q_len: [len(valBatch['vXq'][i]) for i in range(len(valBatch['vX']))],
                    y: valBatch['vY'],
                    keep_prob: 1.0}
            selected_pred = sess.run([prediction], feed_dict=feed_dict)

            for j in range(len(y)):
                print(y[j], selected_pred[j])

if __name__ == "__main__":
    main()
