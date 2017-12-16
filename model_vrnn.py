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

        self.q_z_dim = 50
        self.p_z_dim = 50
        self.p_x_dim = 20
        self.x2s_dim = 10
        self.z2s_dim = 50

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
            eps = tf.random_normal((x.get_shape().as_list()[0], self.z_dim), 0.0, 1.0, dtype=tf.float32)
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




class VRNN():
    def __init__(self, args, sample=False):

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

#                 ss = tf.maximum(1e-10,tf.square(s))
#                 norm = tf.subtract(y[:,:args.chunk_samples], mu)
#                 z = tf.div(tf.square(norm), ss)
#                 denom_log = tf.log(2*np.pi*ss, name='denom_log')
#                 result = tf.reduce_sum(z+denom_log, 1)/2# -
#                                        #(tf.log(tf.maximum(1e-20,rho),name='log_rho')*(1+y[:,args.chunk_samples:])
#                                        # +tf.log(tf.maximum(1e-20,1-rho),name='log_rho_inv')*(1-y[:,args.chunk_samples:]))/2, 1)
# 
#             return result

        def get_likelihood(dec_mu, dec_sigma, dec_rho, dec_binary, y, mask):
            ll = - nllBiGauss(y, dec_mu, dec_sigma, dec_rho, dec_binary)
            return tf.reduce_sum(ll * mask, axis=0)

        # TODO: Need to review KL Divergence formula used here
        def klGaussGauss(mu_1, sigma_1, mu_2, sigma_2):
            with tf.variable_scope("klGaussGauss"):
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

        def get_lossfunc(enc_mu, enc_sigma, dec_mu, dec_sigma, dec_rho, dec_binary, prior_mu, prior_sigma, y, mask):
            kl_loss = klGaussGauss(enc_mu, enc_sigma, prior_mu, prior_sigma)
            nll_loss = nllBiGauss(y, dec_mu, dec_sigma, dec_rho, dec_binary)

            loss = kl_loss + nll_loss
            # import pdb; pdb.set_trace();
            loss = loss * mask
            return tf.reduce_mean(loss, axis=0)
            #return tf.reduce_mean(likelihood_loss)

        self.args = args
        if sample:
            args.batch_size = 1
            args.seq_length = 1

        cell = VariationalRNNCell(args.x_dim, args.rnn_dim, args.z_dim)

        self.cell = cell

        self.input_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.max_length, args.x_dim], name='input_data')
        self.mask = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.max_length], name='mask')

        self.initial_state_c, self.initial_state_h = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

        # input shape: (batch_size, n_steps, n_input)
        # with tf.variable_scope("inputs"):

        # N x t x D
        # sequence_length = tf.shape(self.input_data)[1]
        # t x N x D
        # inputs = tf.transpose(self.input_data, [1, 0, 2])
        # tN x D
        # inputs = tf.reshape(inputs, [sequence_length * args.batch_size, args.x_dim])

        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        # t x (N x D), ie, list of tensors
        # n_steps * (batch_size, n_hidden)
        # inputs = tf.unstack(inputs, axis=0)

        # N x t x D
        inputs = self.input_data

        # Get VRNN cell output
        # Input in batch-major form, the default. No reshaping of input needed!!! :-)
        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs,
                initial_state=tf.nn.rnn_cell.LSTMStateTuple(self.initial_state_c,self.initial_state_h))
                # sequence_length is optional
                # sequence_length=sequence_length, 
                # will provide dtype, no initial state
        # print outputs
        # outputs = map(tf.pack,zip(*outputs))
        outputs_reshape = []
        names = ["enc_mu", "enc_sigma", "dec_mu", "dec_sigma", "dec_rho", "dec_binary", "prior_mu", "prior_sigma"]
        for n,name in enumerate(names):
            with tf.variable_scope(name):
                # N x t x D' (not x_dim)
                x = outputs[n]
                # Nt x D'
                x = tf.reshape(x, [args.batch_size * args.max_length, -1])                
                outputs_reshape.append(x)

        enc_mu, enc_sigma, dec_mu, dec_sigma, dec_rho, dec_binary, prior_mu, prior_sigma = outputs_reshape
        self.final_state_c,self.final_state_h = last_state
        self.mu = dec_mu
        self.sigma = dec_sigma
        self.rho = dec_rho
        self.binary = dec_binary

        # Nt x D
        flat_input = tf.reshape(self.input_data,[args.batch_size * args.max_length, args.x_dim])
        flat_mask = tf.reshape(self.mask, [-1,1])

        # Loss = KL divergence + BiGaussian negative log-likelihood
        lossfunc = get_lossfunc(enc_mu, enc_sigma, dec_mu, dec_sigma, dec_rho, dec_binary, prior_mu, prior_sigma, flat_input, flat_mask)

        # Just get the likelihood (-nll)
        self.likelihood_op = get_likelihood(dec_mu, dec_sigma, dec_rho, dec_binary, flat_input, flat_mask)

        with tf.variable_scope('cost'):
            self.cost = lossfunc 
        tf.summary.scalar('cost', self.cost)
        tf.summary.scalar('mu', tf.reduce_mean(self.mu))
        tf.summary.scalar('sigma', tf.reduce_mean(self.sigma))


        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        for t in tvars:
            print t.name
        grads = tf.gradients(self.cost, tvars)
        #grads = tf.cond(
        #    tf.global_norm(grads) > 1e-20,
        #    lambda: tf.clip_by_global_norm(grads, args.grad_clip)[0],
        #    lambda: grads)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
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
                feed = {self.input_data: prev_x,
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

        for i in xrange(num):
            feed = {self.input_data: prev_x,
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
