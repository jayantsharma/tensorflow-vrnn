from __future__ import print_function
from __future__ import unicode_literals
from builtins import zip
from builtins import range
from builtins import object
import tensorflow as tf
import numpy as np

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

class VariationalRNNCell(tf.contrib.rnn.RNNCell):
    """Variational RNN cell."""

    def __init__(self, x_dim, h_dim, z_dim = 100):
        self.rnn_dim = h_dim
        self.x_dim = x_dim
        self.z_dim = z_dim

        # From the original IAMOnDB model in Theano
        # q_z_dim = 150
        # p_z_dim = 150
        # p_x_dim = 250
        # x2s_dim = 250
        # z2s_dim = 150

        self.q_z_dim = 150
        self.p_z_dim = 150
        self.p_x_dim = 250
        self.x2s_dim = 250
        self.z2s_dim = 150

        self.target_dim = x_dim-1

        self.lstm = tf.contrib.rnn.LSTMCell(self.rnn_dim, state_is_tuple=True)


    # Duck Typing to pass off as an RNN cell
    @property
    def state_size(self):
        return (self.rnn_dim, self.rnn_dim)

    @property
    def output_size(self):
        # works but throws an error down the line
        # return self.z_dim
        # throws an error
        # return self.rnn_dim

        # According to: (enc_mu, enc_sigma, dec_mu, dec_sigma, dec_rho, dec_binary, prior_mu, prior_sigma)
        return (self.z_dim, self.z_dim, self.target_dim, self.target_dim, 1, 1, self.z_dim, self.z_dim)

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # state = (c_state, m_state) == (cell_state, hidden_state)
            c, h = state

            with tf.variable_scope("Prior"):
                with tf.variable_scope("hidden"):
                    prior_hidden = tf.nn.relu(linear(h, self.p_z_dim))
                with tf.variable_scope("mu"):
                    prior_mu = linear(prior_hidden, self.z_dim)
                with tf.variable_scope("sigma"):
                    prior_sigma = tf.nn.softplus(linear(prior_hidden, self.z_dim))

            with tf.variable_scope("phi_x"):
                x_1 = tf.nn.relu(linear(x, self.x2s_dim))

            with tf.variable_scope("Encoder"):
                with tf.variable_scope("hidden"):
                    enc_hidden = tf.nn.relu(linear(tf.concat(axis=1,values=(x_1, h)), self.q_z_dim))
                with tf.variable_scope("mu"):
                    enc_mu = linear(enc_hidden, self.z_dim)
                with tf.variable_scope("sigma"):
                    enc_sigma = tf.nn.softplus(linear(enc_hidden, self.z_dim))

            # The Reparametrization trick
            eps = tf.random_normal((tf.shape(x)[0], self.z_dim), 0.0, 1.0, dtype=tf.float32)
            # z = mu + sigma*epsilon
            z = tf.add(enc_mu, tf.multiply(enc_sigma, eps))
            with tf.variable_scope("phi_z"):
                z_1 = tf.nn.relu(linear(z, self.z2s_dim))

            with tf.variable_scope("Decoder"):
                with tf.variable_scope("hidden"):
                    dec_hidden = tf.nn.relu(linear(tf.concat(axis=1,values=(h, z_1)), self.p_x_dim))
                with tf.variable_scope("mu"):
                    dec_mu = linear(dec_hidden, self.target_dim)
                with tf.variable_scope("sigma"):
                    dec_sigma = tf.nn.softplus(linear(dec_hidden, self.target_dim))
                with tf.variable_scope("rho"):
                    dec_rho = tf.nn.tanh(linear(dec_hidden, 1))
                with tf.variable_scope("binary"):
                    dec_binary = tf.nn.sigmoid(linear(dec_hidden, 1))


            output, state2 = self.lstm(tf.concat(axis=1, values=(x_1, z_1)), state)
        return (enc_mu, enc_sigma, dec_mu, dec_sigma, dec_rho, dec_binary, prior_mu, prior_sigma), state2


def tf_normal(y, mu, s, rho):
    with tf.variable_scope('normal'):
        ss = tf.maximum(1e-10,tf.square(s))
        norm = tf.subtract(y[:,:args.chunk_samples], mu)
        z = tf.div(tf.square(norm), ss)
        denom_log = tf.log(2*np.pi*ss, name='denom_log')
        result = tf.reduce_sum(z+denom_log, 1)/2# -
                               #(tf.log(tf.maximum(1e-20,rho),name='log_rho')*(1+y[:,args.chunk_samples:])
                               # +tf.log(tf.maximum(1e-20,1-rho),name='log_rho_inv')*(1-y[:,args.chunk_samples:]))/2, 1)

    return result

def nllBiGauss(y, mu, sig, corr, binary):
    with tf.variable_scope('nllBiGauss'):
        """
        Bi-Gaussian model negative log-likelihood
        Parameters
        ----------
        y      : Tensor
        mu     : FullyConnected (linear)
        sig    : FullyConnected (softplus)
        corr   : FullyConnected (tanh)
        binary : FullyConnected (sigmoid)
        """
        mu_1 = tf.reshape(mu[:, 0], (-1, 1))
        mu_2 = tf.reshape(mu[:, 1], (-1, 1))

        sig_1 = tf.reshape(sig[:, 0], (-1, 1))
        sig_2 = tf.reshape(sig[:, 1], (-1, 1))

        corr = tf.reshape(corr, (-1, 1))

        y0 = tf.reshape(y[:, 0], (-1, 1))
        y1 = tf.reshape(y[:, 1], (-1, 1))
        y2 = tf.reshape(y[:, 2], (-1, 1))

        c_b = tf.multiply(y0, tf.log(binary)) + tf.multiply(1 - y0, tf.log(1. - binary))

        inner1 =  ((0.5*tf.log(1-corr**2)) +
                   tf.log(sig_1) + tf.log(sig_2) + tf.log(2 * np.pi))

        z = (((y1 - mu_1) / sig_1)**2 + ((y2 - mu_2) / sig_2)**2 -
             (2. * (corr * (y1 - mu_1) * (y2 - mu_2)) / (sig_1 * sig_2)))

        inner2 = 0.5 * (1. / (1. - corr**2))
        cost = - (inner1 + (inner2 * z))

        # nll = -tf.reduce_mean((cost + c_b) * mask, axis=0)   # gives mean
        nll = -(cost + c_b)

    return nll

def likelihood(dec_mu, dec_sigma, dec_rho, dec_binary, y, mask, x_dim):
    # Nt x D
    y_shape = y.shape
    y = tf.reshape(y, [-1, x_dim])

    # mask = tf.reshape(mask, [-1,1])

    ll = - nllBiGauss(y, dec_mu, dec_sigma, dec_rho, dec_binary)
    return tf.reshape(ll * mask, [-1])  # vector

# TODO: Need to review KL Divergence formula used here
def klGaussGauss(mu_1, sigma_1, mu_2, sigma_2):
    with tf.variable_scope("klGaussGauss"):
        # Input:  Nt x D'
        # Output: Nt x 1
        return tf.reduce_sum(0.5 * (
            2 * tf.log(sigma_2,name='log_sigma_2') 
          - 2 * tf.log(sigma_1,name='log_sigma_1')
          + (tf.square(sigma_1) + tf.square(mu_1 - mu_2)) / (tf.square(sigma_2)) - 1
        ), axis=1, keep_dims=True)
#                 return tf.reduce_sum(0.5 * (
#                     2 * tf.log(tf.maximum(1e-9,sigma_2),name='log_sigma_2') 
#                   - 2 * tf.log(tf.maximum(1e-9,sigma_1),name='log_sigma_1')
#                   + (tf.square(sigma_1) + tf.square(mu_1 - mu_2)) / tf.maximum(1e-9,(tf.square(sigma_2))) - 1
#                 ), 1)

def loss(distribution_params, y, mask, x_dim):
    enc_mu, enc_sigma, dec_mu, dec_sigma, dec_rho, dec_binary, prior_mu, prior_sigma = distribution_params

    y_shape = y.shape                  # [N,t,D]
    y = tf.reshape(y, [-1, x_dim])     # Nt x D

    # mask = tf.reshape(mask, [-1,1])

    kl_loss = klGaussGauss(enc_mu, enc_sigma, prior_mu, prior_sigma)
    kl_loss = tf.reshape(kl_loss, [-1, y_shape[1]])             # Nt x 1 -> N x t
    kl_loss = kl_loss * mask

    nll_loss = nllBiGauss(y, dec_mu, dec_sigma, dec_rho, dec_binary)
    nll_loss = tf.reshape(nll_loss, [-1, y_shape[1]])           # Nt x 1 -> N x t
    nll_loss = nll_loss * mask

    loss = kl_loss + nll_loss
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=0))

    # import pdb; pdb.set_trace();
    # loss = loss * mask
    # loss = tf.reshape(loss, [-1])

    return [kl_loss, nll_loss, loss]

def inference(input_data, mask, x_dim, rnn_dim, z_dim):
    cell = VariationalRNNCell(x_dim, rnn_dim, z_dim)
    # input_data = tf.placeholder(dtype=tf.float32, shape=[None, max_length, x_dim], name='input_data')
    # mask = tf.placeholder(dtype=tf.float32, shape=[None, max_length], name='mask')

    initial_state_c, initial_state_h = cell.zero_state(batch_size=tf.shape(input_data)[0], dtype=tf.float32)

    # input shape: (batch_size, n_steps, n_input)
    # with tf.variable_scope("inputs"):

    # N x t x D
    # sequence_length = tf.shape(self.input_data)[1]
    # t x N x D
    # inputs = tf.transpose(self.input_data, [1, 0, 2])
    # tN x D
    # inputs = tf.reshape(inputs, [sequence_length * batch_size, x_dim])

    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    # t x (N x D), ie, list of tensors
    # n_steps * (batch_size, n_hidden)
    # inputs = tf.unstack(inputs, axis=0)

    # N x t x D
    inputs = input_data

    # Get VRNN cell output
    # Input in batch-major form, the default. No reshaping of input needed!!! :-)
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs,
            initial_state=tf.nn.rnn_cell.LSTMStateTuple(initial_state_c,initial_state_h))
            # sequence_length is optional
            # sequence_length=sequence_length, 
            # will provide dtype, no initial state
    # print outputs
    # outputs = map(tf.pack,zip(*outputs))
    outputs_reshape = []
    names = ["enc_mu", "enc_sigma", "dec_mu", "dec_sigma", "dec_rho", "dec_binary", "prior_mu", "prior_sigma"]
    dims = [z_dim]*2 + [x_dim-1]*2 + [1]*2 + [z_dim]*2
    for n,name in enumerate(names):
        with tf.variable_scope(name):
            # N x t x D' (not x_dim)
            x = outputs[n]
            # Nt x D'
            x = tf.reshape(x, [-1, dims[n]])                
            outputs_reshape.append(x)

    # with tf.variable_scope('cost'):
    #     self.cost = lossfunc 
    # tf.summary.scalar('cost', self.cost)
    # tf.summary.scalar('mu', tf.reduce_mean(self.mu))
    # tf.summary.scalar('sigma', tf.reduce_mean(self.sigma))

    return outputs_reshape

def train(loss, lr, global_step):
    tvars = tf.trainable_variables()
    for t in tvars:
        print(t.name)
    optimizer = tf.train.AdamOptimizer(lr)
    grads = optimizer.compute_gradients(loss)
    #grads = tf.cond(
    #    tf.global_norm(grads) > 1e-20,
    #    lambda: tf.clip_by_global_norm(grads, args.grad_clip)[0],
    #    lambda: grads)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)
    return train_op
    #self.saver = tf.train.Saver(tf.all_variables())

def sample(self, sess, args, num=4410, start=None):

    def sample_gaussian(mu, sigma):
        return mu + (sigma*np.random.randn(*sigma.shape))

    if start is None:
        prev_x = np.random.randn(1, 1, 2*args.chunk_samples)
    elif len(start.shape) == 1:
        prev_x = start[np.newaxis,np.newaxis,:]
    elif len(start.shape) == 2:
        for i in range(start.shape[0]-1):
            prev_x = start[i,:]
            prev_x = prev_x[np.newaxis,np.newaxis,:]
            feed = {input_data: prev_x,
                    self.initial_state_c:prev_state[0],
                    self.initial_state_h:prev_state[1]}
            
            [o_mu, o_sigma, o_rho, prev_state_c, prev_state_h] = sess.run(
                    [self.mu, self.sigma, self.rho,
                     self.final_state_c,self.final_state_h],feed)

        prev_x = start[-1,:]
        prev_x = prev_x[np.newaxis,np.newaxis,:]

    prev_state = sess.run(self.cell.zero_state(1, tf.float32))
    chunks = np.zeros((num, 2*args.chunk_samples), dtype=np.float32)
    mus = np.zeros((num, args.chunk_samples), dtype=np.float32)
    sigmas = np.zeros((num, args.chunk_samples), dtype=np.float32)

    for i in range(num):
        feed = {input_data: prev_x,
                self.initial_state_c:prev_state[0],
                self.initial_state_h:prev_state[1]}
        [o_mu, o_sigma, o_rho, next_state_c, next_state_h] = sess.run([self.mu, self.sigma,
            self.rho, self.final_state_c, self.final_state_h],feed)

        next_x = np.hstack((sample_gaussian(o_mu, o_sigma),
                            2.*(o_rho > np.random.random(o_rho.shape[:2]))-1.))
        chunks[i] = next_x
        mus[i] = o_mu
        sigmas[i] = o_sigma

        prev_x = np.zeros((1, 1, 2*args.chunk_samples), dtype=np.float32)
        prev_x[0][0] = next_x
        prev_state = next_state_c, next_state_h

    return chunks, mus, sigmas
