#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np

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
num_epochs = 10000
num_hidden = 50
num_layers = 1
batch_size = 1
initial_learning_rate = 1e-2
momentum = 0.9

num_examples = 2
num_batches_per_epoch = int(num_examples/batch_size)




# Loading the data

audio_filename = maybe_download('red.wav', 96044)
target_filename = maybe_download('red.txt', 12)

audio_filename2=maybe_download('blue.wav', 96044 )
target_filename2 = maybe_download('blue.txt', 19)


fs, audio = wav.read(audio_filename)

fs2,audio2 = wav.read(audio_filename2)

inputs = mfcc(audio, samplerate=fs)
inputs2 = mfcc(audio2, samplerate=fs2)

# Tranform in 3D array
train_inputs = np.asarray(inputs[np.newaxis, :])
#train_inputs = np.asarray(inputs)


train_inputs2 = np.asarray(inputs2[np.newaxis, :])
#train_inputs2 = np.asarray(inputs2)

train_inputs = np.concatenate((train_inputs,train_inputs2))
#train_inputs = np.append(train_inputs,train_inputs2,axis=0)

##print(train_inputs.shape)

#Train_inputs = np.zeros((0,2))
#np.append(Train_inputs,(train_inputs - np.mean(train_inputs))/np.std(train_inputs))
#Train_seq_len = np.zeros((0,2))
train_seq_len = []
#np.append(train_seq_len,[train_inputs.shape[1]])
train_seq_len.append([train_inputs.shape[1]])




#train_inputs2 = np.asarray(inputs2[np.newaxis, :])
#train_inputs2 = (train_inputs2 - np.mean(train_inputs2))/np.std(train_inputs2)
#np.append(Train_inputs,(train_inputs2 - np.mean(train_inputs2))/np.std(train_inputs2))
#np.append(train_seq_len,[train_inputs2.shape[1]]) 
#np.append(Train_seq_len,[train_inputs2.shape[1]])
train_seq_len.append([train_inputs2.shape[1]])

#print (train_seq_len)


#Targets=np.zeros((0,2))
# Readings targets
with open(target_filename, 'r') as f:
    
    #Only the last line is necessary
    line = f.readlines()[-1]    

    # Get only the words between [a-z] and replace period for none
    original = ' '.join(line.strip().lower().split(' ')[2:]).replace('.', '')
    targets = original.replace(' ', '  ')
    targets = targets.split(' ')
    #np.append(Targets,targets)
    print (targets)

with open(target_filename2, 'r') as f:
    
    #Only the last line is necessary
    line = f.readlines()[-1]    

    # Get only the words between [a-z] and replace period for none
    original = ' '.join(line.strip().lower().split(' ')[2:]).replace('.', '')
    targets2 = original.replace(' ', '  ')
    targets2 = targets2.split(' ')

    print ( targets2 )
    #np.append(Targets,targets2)


# Adding blank label
targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])
targets2 = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets2])

# Transform char into index
targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX
                      for x in targets])

targets2 = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX
                      for x in targets2])

# Creating sparse representation to feed the placeholder
#Train_targets = np.zeros(0)
#np.append(Train_targets,sparse_tuple_from([targets]))
#np.append(Train_targets,sparse_tuple_from([targets2]))

train_targets = sparse_tuple_from([targets])
train_targets2 = sparse_tuple_from([targets2])
#print (train_targets)

targets_list = []
#np.asarray(train_targets[np.newaxis, :])
targets_list.append(train_targets)
targets_list.append(train_targets2)
#print (targets_list)

#train_targets2 = np.asarray(train_targets2[np.newaxis, :])
#train_targets = np.concatenate((train_targets,train_targets2))


# We don't have a validation dataset :(
val_inputs, val_targets, val_seq_len = train_inputs, targets_list, \
                                       train_seq_len
#print (Train_inputs) 

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

    #tf.initialize_all_variables().run()
    #session.run(init_op)
    #save_path = saver.save(session, "./orange.ckpt")
    #print("Model saved in file: %s" % save_path)
    saver.restore(session, './orange.ckpt')
    print("Model restored.")


    for curr_epoch in range(num_epochs):
        train_cost = train_ler = 0
        start = time.time()

        for batch in range(num_batches_per_epoch):

	    temp = train_inputs[batch]
            #print (temp.shape)
	    refined_input = np.asarray(temp[np.newaxis, :])
            #print (train_seq_len[batch])
            feed = {inputs: refined_input,
                    targets: targets_list[batch],
                    seq_len: train_seq_len[batch]}
            #print (refined_input.shape)
            batch_cost, _ = session.run([cost, optimizer], feed)
            train_cost += batch_cost*batch_size
            train_ler += session.run(ler, feed_dict=feed)*batch_size

        train_cost /= num_examples
        train_ler /= num_examples

        val_feed = {inputs: refined_input,
                    targets: val_targets[batch],
                    seq_len: val_seq_len[batch]}

        val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)
        


        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
                         val_cost, val_ler, time.time() - start))
    # Decoding
    #print(decoded)
    d = session.run(decoded[0], feed_dict=feed)
    #e = np.asarray(d[1])
    #print (e)
    for i in range(1,3):
    	str_decoded = ''.join([chr(x) for x in np.asarray(d[i]) + FIRST_INDEX])
    	# Replacing blank label to none
    	str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
    	# Replacing space label to space
    	str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
    	#save_path = saver.save(session, "./orange.ckpt")
    	#print("Model saved in file: %s" % save_path)
	print('Original:\n%s' % original)
        print('Decoded:\n%s' % str_decoded)
    x

    

   # d = session.run(decoded[0], feed_dict=feed)
   # dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=session)

   # for i, seq in enumerate(dense_decoded):

    #    seq = [s for s in seq if s != -1]

     #   print('Sequence %d' % i)
     #   print('\t Original:\n%s' % train_targets[i])
     #   print('\t Decoded:\n%s' % seq)
