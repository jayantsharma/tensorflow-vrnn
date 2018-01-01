from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
import numpy as np
import tensorflow as tf

# import argparse
import configargparse
import glob
import time
from datetime import datetime
import os
import pickle

import vrnn

import sys
if not sys.path[0] == '':
    sys.path.insert(0, '')
from nips2015_vrnn.datasets.iamondb import IAMOnDB

from matplotlib import pyplot as plt

'''
TODOS:
    - parameters for depth and width of hidden layers
    - implement predict function
    - separate binary and gaussian variables
    - clean up nomenclature to remove MDCT references
    - implement separate MDCT training and sampling version
'''

def load_datasets(FLAGS):
    training_data = IAMOnDB(name='train',
                         prep='normalize',
                         cond=False,
                         path=FLAGS.data_path).data
    validation_data = IAMOnDB(name='valid',
                         prep='normalize',
                         cond=False,
                         path=FLAGS.data_path).data

    """
    Data (training/validation) is a vector of size num_samples, in which each of the samples is sized T x 3.
    np.array(data) normally converts nested lists to ndarrays, but is not possible here because each of the 
    samples is of different size (diff num of samples in series).
    
    Use a generator function to iterate over each of the samples.
    """
    def gen_train_samples():
        data = training_data[0]
        for i in range(data.shape[0]):
            yield data[i]
    def gen_valid_samples():
        data = validation_data[0]
        for i in range(data.shape[0]):
            yield data[i]

    training_dataset = tf.data.Dataset.from_generator(gen_train_samples, tf.float32, tf.TensorShape([None,3]))
    validation_dataset = tf.data.Dataset.from_generator(gen_valid_samples, tf.float32, tf.TensorShape([None,3]))

    # Indicates number of timesteps in each sequence, since we'll zero-pad each sequence to max_sequence_length.
    def build_mask(x):
        # x.shape = T x 3
        return tf.ones_like(x[:,0])
    training_dataset = training_dataset.map(lambda x: (x, build_mask(x)))
    validation_dataset = validation_dataset.map(lambda x: (x, build_mask(x)))

    # Every sequence has shape: length x features
    training_dataset = training_dataset.padded_batch(FLAGS.batch_size, 
            padded_shapes=([FLAGS.max_length, FLAGS.x_dim], [FLAGS.max_length]))
    validation_dataset = validation_dataset.padded_batch(FLAGS.validation_size, 
            padded_shapes=([FLAGS.max_length, FLAGS.x_dim], [FLAGS.max_length]))

    training_dataset = training_dataset.repeat(FLAGS.num_epochs)
    validation_dataset = validation_dataset.repeat()

    return training_dataset, validation_dataset


def process_dataset():
    # Set up data pipeline
    training_dataset, validation_dataset = load_datasets(FLAGS)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
            handle, training_dataset.output_types, training_dataset.output_shapes)
    input, mask = iterator.get_next()

    training_iterator = training_dataset.make_one_shot_iterator()
    validation_iterator = validation_dataset.make_one_shot_iterator()
    # Training input
    # t_input, t_mask = training_iterator.get_next()
    # Validation input
    # v_input, v_mask = validation_iterator.get_next()

    return training_iterator, validation_iterator, input, mask


# def get_likelihood(input, mask, FLAGS):
#     _, _, dec_mu, dec_sigma, dec_rho, dec_binary, _, _ = vrnn.inference(input, mask, FLAGS.x_dim, 
#                                                                         FLAGS.rnn_dim, FLAGS.z_dim)
#     likelihood = vrnn.likelihood(dec_mu, dec_sigma, dec_rho, dec_binary, input, mask, FLAGS.x_dim)
#     return likelihood


def train(FLAGS):
    dirname = 'save-vrnn'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, 'config.pkl'), 'wb') as f:
        pickle.dump(FLAGS, f)

    global_step = tf.train.get_or_create_global_step()

    # Get training and validation inputs
    training_iterator, validation_iterator, input, mask = process_dataset()

    distribution_params = vrnn.inference(input, mask, FLAGS.x_dim, FLAGS.rnn_dim, FLAGS.z_dim)

    # Loss = KL divergence + BiGaussian negative log-likelihood
    loss = vrnn.loss(distribution_params, input, mask, FLAGS.x_dim)
    training_op = vrnn.train(loss, FLAGS.lr, global_step)

    # ll = -nll
    _, _, dec_mu, dec_sigma, dec_rho, dec_binary, _, _ = distribution_params
    likelihood = vrnn.likelihood(dec_mu, dec_sigma, dec_rho, dec_binary, input, mask, FLAGS.x_dim)

    # training_init_op = iterator.make_initializer(training_dataset)
    # validation_init_op = iterator.make_initializer(validation_dataset)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        if self._step % FLAGS.monitor_every == 0:
            return tf.train.SessionRunArgs([distribution_params, likelihood], 
                                            feed_dict={handle: validation_handle})

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.monitor_every == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          _, likelihood_value = run_values.results
          examples_per_sec = FLAGS.monitor_every * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.monitor_every)

          format_str = ('%s: Batches %d, likelihood_lower_bound = %.2f (%.1f examples/sec; %.3f sec/batch)')
          print (format_str % (datetime.now(), self._step, likelihood_value,
                               examples_per_sec, sec_per_batch))
          print ('--'*20 + '\n' + '--'*20)


    ckpt = tf.train.get_checkpoint_state(dirname)
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=dirname,
        hooks=[_LoggerHook()]) as mon_sess:

        # Debugger Needs
        # with tf_debug.LocalCLIDebugWrapperSession(mon_sess) as sess:

        training_handle = mon_sess.run(training_iterator.string_handle())
        validation_handle = mon_sess.run(validation_iterator.string_handle())

        while not mon_sess.should_stop():
          mon_sess.run(training_op, feed_dict={handle: training_handle})
        _, likelihood = mon_sess.run([distribution_params, likelihood], feed_dict={handle: validation_handle})
    print ('Training finished.')
    format_str = ('likelihood_lower_bound = %.2f')
    print (format_str % (likelihood))
        # summary_writer = tf.summary.FileWriter('logs/' + datetime.now().isoformat().replace(':', '-'), sess.graph)
        # check = tf.add_check_numerics_ops()
        # merged = tf.summary.merge_all()

        # tf.global_variables_initializer().run()
        # saver = tf.train.Saver(tf.global_variables())
        # if ckpt:
            # saver.restore(sess, ckpt.model_checkpoint_path)
            # print("Loaded model")

        # for e in range(1, FLAGS.num_epochs+1):
            # print('Processing epoch: {}'.format(e))
            # sess.run(training_init_op)
            # sess.run(tf.assign(model.lr, FLAGS.lr))
            # sess.run(tf.assign(model.lr, FLAGS.lr * (FLAGS.decay_rate ** e)))
            # state = model.initial_state_c, model.initial_state_h
            # b = 1
            # while True:
                # try:
                    # print('Training batch : {}'.format(b))
                    # sess.run(training_op)
                    # if b % 20 == 0:
                        # print('Time take for last 100 batches: {}'.format(time.time() - st))
                        # st = time.time()
                    # b += 1
                # except tf.errors.OutOfRangeError:
                    # break

            # end-of-epoch processing
            # LOG LIKELIHOOD EVERY m EPOCHS
            # if e % FLAGS.monitor_every == 0:
                # sess.run(validation_init_op)
                # ll = 0
                # b = 1
                # while True:
                    # try:
                        # print('Validation batch : {}'.format(b))
                        # ll += sess.run(likelihood)
                        # b += 1
                    # except tf.errors.OutOfRangeError:
                        # print("{}/{} (epoch {}), log_likelihood = {}".format(e, FLAGS.num_epochs, e, ll))
                        # summary_writer.add_summary(summary, e)
                        # if e % FLAGS.save_every == 0:
                            # checkpoint_path = os.path.join(dirname, 'model.ckpt')
                            # saver.save(sess, checkpoint_path, global_step=e)
                            # print("model saved to {}".format(checkpoint_path))
                        # break

if __name__ == '__main__':
    p = configargparse.ArgParser(default_config_files=['tensorflow-vrnn/iamondb.conf'])
    p.add('-c', '--my-config',
            is_config_file=True, help='config file path')
    p.add('--data_path',  help='Data path')
    p.add('--max_length', type=int, default=2000, help='maximum sequence length')
    p.add('--rnn_dim',    type=int, default=3, help='size of RNN hidden state')
    p.add('--x_dim',      type=int, default=3, help='size of input')
    p.add('--z_dim',      type=int, default=3, help='size of latent space')
    p.add('--num_k',      type=int, default=20, help='number of GMM components')
    p.add('--batch_size', type=int, default=3000, help='minibatch size')
    p.add('--validation_size', type=int, default=1438, help='validation data size')
    p.add('--num_epochs', type=int, default=100, help='number of epochs')
    p.add('--lr',         type=float, default=0.0005, help='learning rate')
    p.add('--decay_rate', type=float, default=1., help='decay of learning rate')
    p.add('--save_every', type=int, default=500, help='save frequency')
    p.add('--monitor_every', type=int, default=10, help='monitoring frequency')
    p.add('--grad_clip',  type=float, default=10., help='clip gradients at this value')
    p.add('--debug',      type=int, default=0, help='debug')
    # p.add('-d', '--dbsnp', help='known variants .vcf', env_var='DBSNP_PATH')

    FLAGS = p.parse_args()

    train(FLAGS)
