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

def next_batch(args):
    t0 = np.random.randn(args.batch_size, 1, (2 * args.chunk_samples))
    mixed_noise = np.random.randn(
        args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1
    #x = t0 + mixed_noise + np.random.randn(
    #    args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1
    #y = t0 + mixed_noise + np.random.randn(
    #    args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1
    x = np.sin(2 * np.pi * (np.arange(args.seq_length)[np.newaxis, :, np.newaxis] / 10. + t0)) + np.random.randn(
        args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1 + mixed_noise*0.1
    y = np.sin(2 * np.pi * (np.arange(1, args.seq_length + 1)[np.newaxis, :, np.newaxis] / 10. + t0)) + np.random.randn(
        args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1 + mixed_noise*0.1

    y[:, :, args.chunk_samples:] = 0.
    x[:, :, args.chunk_samples:] = 0.
    return x, y

def load_datasets(args):
    training_data = IAMOnDB(name='train',
                         prep='normalize',
                         cond=False,
                         path=args.data_path).data
    validation_data = IAMOnDB(name='valid',
                         prep='normalize',
                         cond=False,
                         path=args.data_path).data

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
    training_dataset = training_dataset.padded_batch(args.batch_size, 
            padded_shapes=([args.max_length, args.x_dim], [args.max_length]))
    validation_dataset = validation_dataset.padded_batch(args.batch_size, 
            padded_shapes=([args.max_length, args.x_dim], [args.max_length]))
    return training_dataset, validation_dataset

def train(args):
    dirname = 'save-vrnn'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    ckpt = tf.train.get_checkpoint_state(dirname)
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('logs/' + datetime.now().isoformat().replace(':', '-'), sess.graph)
        # check = tf.add_check_numerics_ops()
        # merged = tf.summary.merge_all()

        # Set up data pipeline
        training_dataset, validation_dataset = load_datasets(args)

        iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
        input_data, mask = iterator.get_next()

        distribution_params = vrnn.inference(input_data, mask, args.x_dim, args.rnn_dim, args.z_dim)

        # Loss = KL divergence + BiGaussian negative log-likelihood
        loss = vrnn.loss(distribution_params, input_data, mask, args.x_dim)
        train_op = vrnn.train(loss, args.lr)

        # ll = -nll
        _, _, dec_mu, dec_sigma, dec_rho, dec_binary, _, _ = distribution_params
        likelihood_op = vrnn.likelihood(dec_mu, dec_sigma, dec_rho, dec_binary, input_data, mask, args.x_dim)

        # import ipdb; ipdb.set_trace(context=5);

        training_init_op = iterator.make_initializer(training_dataset)
        validation_init_op = iterator.make_initializer(validation_dataset)

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Loaded model")

        for e in range(1, args.num_epochs+1):
            print('Processing epoch: {}'.format(e))
            sess.run(training_init_op)
            # sess.run(tf.assign(model.lr, args.lr))
            # sess.run(tf.assign(model.lr, args.lr * (args.decay_rate ** e)))
            # state = model.initial_state_c, model.initial_state_h
            b = 1
            while True:
                try:
                    print('Training batch : {}'.format(b))
                    sess.run(train_op)
                    b += 1
                except tf.errors.OutOfRangeError:
                    break

            # end-of-epoch processing
            # LOG LIKELIHOOD EVERY m EPOCHS
            if e % args.monitor_every == 0:
                sess.run(validation_init_op)
                ll = 0
                b = 1
                while True:
                    try:
                        print('Validation batch : {}'.format(b))
                        ll += sess.run(likelihood_op)
                        b += 1
                    except tf.errors.OutOfRangeError:
                        print("{}/{} (epoch {}), log_likelihood = {}".format(e, args.num_epochs, e, ll))
                        # summary_writer.add_summary(summary, e)
                        if e % args.save_every == 0:
                            checkpoint_path = os.path.join(dirname, 'model.ckpt')
                            saver.save(sess, checkpoint_path, global_step=e)
                            print("model saved to {}".format(checkpoint_path))
                        break

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
    p.add('--num_epochs', type=int, default=100, help='number of epochs')
    p.add('--lr',         type=float, default=0.0005, help='learning rate')
    p.add('--decay_rate', type=float, default=1., help='decay of learning rate')
    p.add('--save_every', type=int, default=500, help='save frequency')
    p.add('--monitor_every', type=int, default=10, help='monitoring frequency')
    p.add('--grad_clip',  type=float, default=10., help='clip gradients at this value')
    p.add('--debug',      type=int, default=0, help='debug')
    # p.add('-d', '--dbsnp', help='known variants .vcf', env_var='DBSNP_PATH')

    args = p.parse_args()

    train(args)
