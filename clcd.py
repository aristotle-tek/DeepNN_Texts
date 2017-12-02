#

'''
CNN LSTM Model for regression with concrete dropout (Gal, et al 2017)
	- generates simulated speech data online (but uses fixed test data).
	- depends: concrete_fns.py and ideol_utils.py.
'''



from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Input, InputLayer, Flatten, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
#from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint
#from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras import initializers
from keras.engine import InputSpec
from keras.layers import Dense, Lambda, Wrapper
from keras.layers.merge import concatenate
from keras.callbacks import Callback, CSVLogger

import h5py
import time
import cPickle as pickle
import numpy as np
import pandas as pd

import os, sys
from glob import glob
import fnmatch
import os.path
import re
from collections import Counter
from collections import deque
import codecs

import logging
from datetime import date
from datetime import datetime
import pickle as pickle

from numpy.random import multinomial
from numpy.random import binomial
from numpy.random import normal

from itertools import product

from concrete_fns import ConcreteDropout, logsumexp

from ideol_utils import *

import argparse


parser = argparse.ArgumentParser(description='Ideological sentence analysis.')
parser.add_argument('-n','--modelname', help='name', type=str, required=False, default='unkmodel')
parser.add_argument('-l','--slen', help='words per sentence', type=int, required=False, default=80)
parser.add_argument('-f','--nfunc', help='num of possible functions', type=int, required=False, default=4)
parser.add_argument('-r','--randorder', help='randomize sentence order', type=bool, required=False, default=True)
parser.add_argument('-p','--pctnoise', help='pct_noise_sentences', type=float, required=False, default=0)
#parser.add_argument('-s','--speeches', help='numspeeches_per_speaker', type=int, required=False, default=10)
parser.add_argument('-x','--fitsteps', help='steps per epoch', type=int, required=False, default=10000)
parser.add_argument('-k','--mcsamples', help='num mc samples', type=int, required=False, default=1000)
parser.add_argument('-e','--epochs', help='num epochs', type=int, required=False, default=10)
parser.add_argument('-b','--batchsize', help='batch size', type=int, required=False, default=28)

parser.add_argument('-c','--concretedrop', help='whether use concrete dropout', type=int, required=False, default=True)
parser.add_argument('-s','--speechdrawideal', help='whether draw speech ideal points from distrib', type=bool, required=False, default=True)
parser.add_argument('-v','--speechdispers', help='dispersion of speeches around ideal pt', type=int, required=False, default=6)
parser.add_argument('-o','--folder', help='folder string', type=str, required=False, default='alpha')


args = vars(parser.parse_args())


for k, v in args.iteritems():
	print("%s : %s" % (str(k), str(v),) )

name = args['modelname']
model_id = name

sent_length = args['slen']
numfunc = args['nfunc']
randorder = args['randorder']
pct_noise_sentences = args['pctnoise']
#speeches_per_spkr = args['speeches']
K_test = args['mcsamples']
fitsteps = args['fitsteps']
concretedrop = args['concretedrop']
if concretedrop ==0:
	concretedrop=False
else:
	concretedrop=True

speech_draw_ideal = args['speechdrawideal']
speechdispersion = args['speechdispers']

num_partisan_words = 1000
num_speakers = 200
leadership_speak_ratio = .5 # percent of speeches made by leadership members
pct_leadership = .3 # percent of speakers who are members of the leadership


# Training
epochs = args['epochs']
batch_size= args['batchsize']


ckpt_period = 1

folderstring= args['folder']

wtspath = "/out_" +folderstring + "/" + 'chkpt_' + model_id + "e{epoch:02d}"+ ".h5"
datapath = "/" +folderstring  +"/"
outpath = "/out_" +folderstring + "/"

if not os.path.exists(wtspath):
	os.makedirs(wtspath)

if not os.path.exists(datapath):
	os.makedirs(datapath)

if not os.path.exists(outpath):
	os.makedirs(outpath)


# Embedding
max_features = 20000
maxlen = sent_length
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

#- concrete:
l = 1e-4
D = 1
K_test = 20 # for mc samples.


#----------------------------
print('Build model...')
#----------------------------

N = epochs * fitsteps
wd = l**2. / N
dd = 2. / N


if concretedrop:
	embedding_layer = Embedding(max_features, embedding_size, input_length=maxlen, trainable=True)
	sequence_input = Input(shape=(maxlen,), dtype='int32')
	embedded_sequences = embedding_layer(sequence_input)
	x = ConcreteDropout(Conv1D(filters, kernel_size, padding='valid', 
		strides=1, activation='relu'), weight_regularizer=wd, 
		dropout_regularizer=dd)(embedded_sequences)
	x = MaxPooling1D(pool_size=pool_size)(x)
	x = ConcreteDropout(LSTM(lstm_output_size), weight_regularizer=wd, dropout_regularizer=dd)(x)
	preds = ConcreteDropout(Dense(1, kernel_initializer='normal'),weight_regularizer=wd, dropout_regularizer=dd)(x)
	model = Model(sequence_input, preds)
	model.compile(optimizer='adam', loss='mean_squared_error')
else:
	embedding_layer = Embedding(max_features, embedding_size, input_length=maxlen, trainable=True)
	sequence_input = Input(shape=(maxlen,), dtype='int32')
	embedded_sequences = embedding_layer(sequence_input)
	x = Conv1D(filters, kernel_size, padding='valid', 
		strides=1, activation='relu')(embedded_sequences)
	x = MaxPooling1D(pool_size=pool_size)(x)
	x = LSTM(lstm_output_size)(x)
	preds = Dense(1, kernel_initializer='normal')(x)
	model = Model(sequence_input, preds)
	model.compile(optimizer='adam', loss='mean_squared_error')



#------------------------------------
# loss history
#------------------------------------

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
#------------------------------------

#------------------------------------
history = LossHistory()

csv_logger = CSVLogger(outpath + 'training' + model_id +'.log')


checkpoint = ModelCheckpoint(wtspath, verbose=2, save_best_only=False, period=ckpt_period)

print(model.summary())
#config = model.get_config()
#pickle.dump(config, open("config_" + model_id + ".pkl", "wb" ) )

#------------------------------------
#    Prep data
#------------------------------------
print("loading data...")

vocab_lists = pickle.load(open(datapath + "vocab_lists.pkl", "rb" ) )
df = pd.read_csv(datapath + "speaker_df.csv")


leader_ids = df.memid[df.leader==1]
nonleader_ids= df.memid[df.leader==0]

partisan_words_dict, nonpartisan_words, func_words = vocab_lists[1], vocab_lists[2],vocab_lists[3]

online_batch_gen = gen_batches_online(batch_size, df, leader_ids,
	nonleader_ids, leadership_speak_ratio, sent_length, nonpartisan_words,
 	func_words, partisan_words_dict, pct_noise_sentences, 
	speech_draw_ideal=speech_draw_ideal, speechdispersion=speechdispersion )

# fixed data for validation, testing:
testfile = datapath + "ideal_0.csv"
x_test, y_test = load_from_csv(testfile, sent_length, shuffle=False)

#------------------------------------

#------------------------------------
tick = time.time()
print('Training...')

model.fit_generator(generator=online_batch_gen, steps_per_epoch=fitsteps,
	epochs=epochs, verbose=2, callbacks=[history, csv_logger, checkpoint])

print(history.losses[-5:])

traintime = (time.time() - tick)/60.0
print("fit time: %.1f mins" % traintime)

mse = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print('Loss: ', mse)


pr = model.predict(x_test, batch_size=batch_size, verbose=0)
print("variance: %.3f" % np.var(pr))

testset = x_test[:1000,:]
Y_true = y_test[:1000]

MC_samples = np.array([model.predict(testset) for _ in range(K_test)])

pred_savefile = outpath + "preds_" + model_id + ".txt"
np.savetxt(pred_savefile, MC_samples)


lossfile = outpath + "losshist_" + model_id + ".txt" 
np.savetxt(lossfile, history.losses)


print("\n\ndone.\n")
