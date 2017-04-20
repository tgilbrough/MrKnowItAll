import argparse
import tensorflow as tf
from tqdm import tqdm
import os

from baseline import Model
from data import Data

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep_prob', '-kp', type=float, default=0.7)
    parser.add_argument('--hidden_size', '-hs', type=int, default=100)
    parser.add_argument('--emb_size', '-es', type=int, default=50) # this could be 50 (171.4 MB), 100 (347.1 MB), 200 (693.4 MB), or 300 (1 GB)
    parser.add_argument('--train_path', default='./datasets/msmarco/train/location.json')
    parser.add_argument('--val_path', default='./datasets/msmarco/dev/location.json')
    parser.add_argument('--reference_path', default='./eval/references.json')
    parser.add_argument('--candidate_path', default='./eval/candidates.json')
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01)
    parser.add_argument('--num_threads', '-t', type=int, default=4)
    parser.add_argument('--model_save_dir', default='./saved_models')

    return parser

def main():
    parser = get_parser()
    config = parser.parse_args()

    tf.reset_default_graph()

    data = Data(config)
    model = Model(config, data.max_context_size, data.max_ques_size)

    tensorboard_path = './tensorboard_models/' + model.model_name
    save_model_path = config.model_save_dir + '/' + model.model_name 
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    print('Building tensorflow computation graph...')

    # Data input queues
    train_queue, train_qr = data.getTrainQueueRunner()
    x, q, y_begin, y_end = train_queue.dequeue_many(config.batch_size)

    keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

    x_len = [data.max_context_size for i in range(config.batch_size)]
    q_len = [data.max_ques_size for i in range(config.batch_size)]

    outputs = model.build(x, x_len, q, q_len, data.embeddings, keep_prob)

    print('Computation graph completed.')

    with tf.variable_scope('loss'):
        logits1, logits2 = outputs['logits_start'], outputs['logits_end']
        loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_begin, logits=logits1), name='beginning_loss')
        loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_end, logits=logits2), name='ending_loss')
        loss = loss1 + loss2
    with tf.variable_scope('accuracy'):
        acc1 = tf.reduce_mean(tf.cast(tf.equal(y_begin, tf.cast(tf.argmax(logits1, 1), 'int32')), 'float'), name='beginning_accuracy')
        acc2 = tf.reduce_mean(tf.cast(tf.equal(y_end, tf.cast(tf.argmax(logits2, 1), 'int32')), 'float'), name='ending_accuracy')

    train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(loss)
    
    number_of_train_batches = data.getNumTrainBatches()
    number_of_val_batches = data.getNumValBatches()

    # For tensorboard
    train_writer = tf.summary.FileWriter(tensorboard_path + '/train')
    val_writer = tf.summary.FileWriter(tensorboard_path + '/dev')
    
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("loss1", loss1)
    tf.summary.scalar("loss2", loss2)
    tf.summary.scalar("accuracy1", acc1)
    tf.summary.scalar("accuracy2", acc2)
    merged_summary = tf.summary.merge_all()

    # For saving models
    saver = tf.train.Saver()
    min_val_loss = float('Inf')

    with tf.Session() as sess:
        train_writer.add_graph(sess.graph)
        val_writer.add_graph(sess.graph)

        coord = tf.train.Coordinator()
        enqueue_threads = train_qr.create_threads(sess, coord=coord, start=True)
        
        sess.run(tf.global_variables_initializer())

        for e in range(config.epochs):
            print('Epoch {}/{}'.format(e + 1, config.epochs))
            for i in tqdm(range(number_of_train_batches)):
                if coord.should_stop():
                    break

                sess.run(train_step, feed_dict={keep_prob: config.keep_prob})
                
                if (e * number_of_train_batches + i) % 20 == 0:
                    # Record results for tensorboard
                    train_sum = sess.run(merged_summary, feed_dict={keep_prob: config.keep_prob})

                    valBatch = data.getValBatch()
                    val_sum, val_loss  = sess.run([merged_summary, loss], feed_dict={x: valBatch['vX'],
                                                                    q: valBatch['vXq'],
                                                                    y_begin: valBatch['vYBegin'],
                                                                    y_end: valBatch['vYEnd'],
                                                                    keep_prob: 1.0})
                    if val_loss < min_val_loss:
                        saver.save(sess, save_model_path + '/model')
                        min_val_loss = val_loss

                    train_writer.add_summary(train_sum, e * number_of_train_batches + i)
                    val_writer.add_summary(val_sum, e * number_of_train_batches + i)

        coord.request_stop()
        coord.join(enqueue_threads)

        # Load best graph on validation data
        
        new_saver = tf.train.import_meta_graph(save_model_path + '/model.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(save_model_path))
        all_vars = tf.get_collection('vars')
        for v in all_vars:
            v_ = sess.run(v)
            print(v_)

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
            batch = data.getValBatch()

            prediction_begin = tf.cast(tf.argmax(logits1, 1), 'int32')
            prediction_end = tf.cast(tf.argmax(logits2, 1), 'int32')

            begin, end = sess.run([prediction_begin, prediction_end], feed_dict={x: batch['vX'],
                                                                                q: batch['vXq'],
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

                begin_corr += int(begin[j] == batch['vYBegin'][j])
                end_corr += int(end[j] == batch['vYEnd'][j])
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