from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
import tensorflow as tf

import os
import pickle
from model_vrnn import VRNN
import numpy as np

from train_vrnn import next_batch

with open(os.path.join('save-vrnn', 'config.pkl')) as f:
    saved_args = pickle.load(f)

model = VRNN(saved_args, True)
sess = tf.InteractiveSession()
saver = tf.train.Saver(tf.all_variables())

ckpt = tf.train.get_checkpoint_state('save-vrnn')
print("loading model: ",ckpt.model_checkpoint_path)

saver.restore(sess, ckpt.model_checkpoint_path)
sample_data,mus,sigmas = model.sample(sess,saved_args)
