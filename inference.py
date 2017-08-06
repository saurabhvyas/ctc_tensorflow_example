#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np
import os

from six.moves import xrange as range

try:
    from tensorflow.python.ops import ctc_ops
except ImportError:
    from tensorflow.contrib.ctc import ctc_ops

try:
    from python_speech_features import mfcc
except ImportError:
    print("Failed to import python_speech_features.\n Try pip install python_speech_features.")
    raise ImportError

from utils import maybe_download as maybe_download
from utils import sparse_tuple_from as sparse_tuple_from

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

# Some configs
num_features = 13
# Accounting the 0th indice +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters
num_epochs = 1000
num_hidden = 50
num_layers = 1
batch_size = 1
initial_learning_rate = 1e-2
momentum = 0.9


fs, audio = wav.read('/home/saurabh/ctc_tensorflow_example/color2.wav')

inputs = mfcc(audio, samplerate=fs)
# Tranform in 3D array
train_inputs = np.asarray(inputs[np.newaxis, :])
train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)
train_seq_len = [train_inputs.shape[1]]


# Loading the data
from os import listdir
from os.path import isfile, join


#print (train_seq_len) 

# THE MAIN CODE!

graph = tf.Graph()
with graph.as_default():
    # e.g: log filter bank or MFCC features
    # Has size [batch_size, max_stepsize, num_features], but the
    # batch_size and max_stepsize can vary along each step
    inputs = tf.placeholder(tf.float32, [None, None, num_features])

    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32)

    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])

    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, state_is_tuple=True)

    # Stacking rnn cells
    stack =  tf.contrib.rnn.MultiRNNCell([cell] * num_layers,
                                        state_is_tuple=True)

    # The second output is the last state and we will no use that
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1))
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    loss = ctc_ops.ctc_loss( targets, logits , seq_len)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.MomentumOptimizer(initial_learning_rate,
                                           0.9).minimize(cost)

    # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = ctc_ops.ctc_greedy_decoder(logits, seq_len)

    # Inaccuracy: label error rate
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targets))

    saver = tf.train.Saver()


    # Decoding
    
   # print(train_inputs.values()[0].shape)
   # train_inputs2=np.append(train_inputs.values(), axis=1)
   # print(type(train_inputs2))

    
   # print (num_examples)
   # print(d)
    
    with tf.Session(graph=graph) as session:
    # Initializate the weights and biases
    #init_op = tf.global_variables_initializer()

    #init_op.run()
      saver.restore(session, './orange3.ckpt')
      print("Model restored.")
    
        
      feed2 = {inputs: train_inputs,
                 seq_len: train_seq_len}
      d = session.run(decoded[0], feed_dict=feed2)
        #print (d)
      str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
    	# Replacing blank label to none
      str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
    	# Replacing space label to space
      str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
    	
	#print('Original:\n%s' % )
      print('Decoded:\n%s' % str_decoded)
    #save_path = saver.save(session, "./orange3.ckpt")
    #print("Model saved in file: %s" % save_path)
    x

   
