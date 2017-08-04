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
num_epochs = 10
num_hidden = 50
num_layers = 1
batch_size = 1
initial_learning_rate = 1e-2
momentum = 0.9





# Loading the data
from os import listdir
from os.path import isfile, join

mypath='/home/saurabh/ctc_tensorflow_example/data'
data_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
txt_files = [ fi for fi in data_files if not fi.endswith(".wav") ]
txt_files = lst = [os.path.splitext(x)[0] for x in txt_files]
wav_files = [ fi for fi in data_files if not fi.endswith(".txt") ]
wav_files = lst = [os.path.splitext(x)[0] for x in wav_files]

num_examples = len(wav_files)
print ("number of data examples : " + str(num_examples) )
num_batches_per_epoch = int(num_examples/batch_size)


#print (wav_files)
audio_filename={}
target_filename={}
fs={}
audio={}
inputs={}
train_inputs={}
train_seq_len = {}
targets={}
train_targets={}
original={}


for i,j in enumerate(wav_files):
        #print (i,j)
	audio_filename[i] =  '/home/saurabh/ctc_tensorflow_example/data/' + j + '.wav'
	#print ( audio_filename[i])
	#print (audio_filename)
	target_filename[i] =  '/home/saurabh/ctc_tensorflow_example/data/' + j + '.txt'
	fs[i], audio[i] = wav.read( audio_filename[i])
	#print (audio[i])

	
	#print (temp2)
	inputs[i] = mfcc(audio[i], samplerate=fs[i])
	#print ( inputs[i].shape , fs[i] )
	temp=inputs[i]
	# Tranform in 3D array
	
	train_inputs[i] = np.asarray(temp[np.newaxis, :])
	train_seq_len[i]=[train_inputs[i].shape[1]]
	#print ("new shape " + str(train_inputs[i].shape))
	with open(target_filename[i], 'r') as f:
    
   	 #Only the last line is necessary
    		line = f.readlines()[-1]    

   	 # Get only the words between [a-z] and replace period for none
    		original[i] = ' '.join(line.strip().lower().split(' ')).replace('.', '')
    		targets[i] = original[i].replace(' ', '  ')
    		targets[i] = targets[i].split(' ')
    	#np.append(Targets,targets)

	# Adding blank label
		targets[i] = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets[i]])


	# Transform char into index
		targets[i] = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX
                      for x in targets[i]])
	


	# Creating sparse representation to feed the placeholder

		train_targets[i] = sparse_tuple_from([targets[i]])


#train_inputs = np.concatenate(tuple(train_inputs.values()),axis=1)
#print (len(train_inputs))




#Targets=np.zeros((0,2))
# Readings targets


#targets_list = []
#np.asarray(train_targets[np.newaxis, :])
#targets_list.append(train_targets)
#targets_list.append(train_targets2)
#print (targets_list)

#train_targets2 = np.asarray(train_targets2[np.newaxis, :])
#train_targets = np.concatenate((train_targets,train_targets2))


# We don't have a validation dataset :(
val_inputs, val_targets, val_seq_len = train_inputs, train_targets, \
                                       train_seq_len
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

with tf.Session(graph=graph) as session:
    # Initializate the weights and biases
    init_op = tf.global_variables_initializer()

    init_op.run()
   # saver.restore(session, './orange.ckpt')
   # print("Model restored.")


    for curr_epoch in range(num_epochs):
        train_cost = train_ler = 0
        start = time.time()

        for batch in range(num_batches_per_epoch):

            
	    temp = train_inputs[batch]
            #print (train_targets)
	   # print ( train_seq_len[batch] )
	    #refined_input = np.asarray(temp[np.newaxis, :])
          #  print (train_seq_len[batch])
            feed = {inputs: temp,
                    targets: train_targets[batch],
                    seq_len: train_seq_len[batch]}
            #print (refined_input.shape)
            batch_cost, _ = session.run([cost, optimizer], feed)
            train_cost += batch_cost*batch_size
            train_ler += session.run(ler, feed_dict=feed)*batch_size

        train_cost /= num_examples
        train_ler /= num_examples

        val_feed = {inputs: temp,
                    targets:train_targets[batch] ,
                    seq_len: train_seq_len[batch]}

        val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)
        


        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
                         val_cost, val_ler, time.time() - start))
    # Decoding
    
   # print(train_inputs.values()[0].shape)
   # train_inputs2=np.append(train_inputs.values(), axis=1)
   # print(type(train_inputs2))

    
   # print (num_examples)
   # print(d)
    
    for i in range(1, num_examples):

        
        feed2 = {inputs: train_inputs[i-1],
                    targets: train_targets[i-1],
                    seq_len: train_seq_len[i-1]}
        d = session.run(decoded[0], feed_dict=feed2)
        #print (d)
    	str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
    	# Replacing blank label to none
    	str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
    	# Replacing space label to space
    	str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
    	save_path = saver.save(session, "./orange.ckpt")
    	print("Model saved in file: %s" % save_path)
	print('Original:\n%s' % original[i-1])
        print('Decoded:\n%s' % str_decoded)

    x

   
