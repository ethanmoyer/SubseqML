# Copyright 2019 Hannes Bretschneider
# 
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

import tensorflow as tf
# import theano
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Activation, Lambda, Bidirectional, TimeDistributed, Flatten, Reshape
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
import os

class Entry:
	def __init__(self, gene, seq, seqs, contains, gap, op, b = 0):
		self.gene = gene
		self.seq = seq
		self.seqs = seqs
		self.contains = contains
		self.gap = gap
		self.b = b
		self.op = op


def create_entries():
	mypath = 'data2/'
	from os import listdir
	from os.path import isfile, join
	files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	entries = []
	for file in files[:10]:
		split_file = file.split('_')
		print(split_file)
		with open(mypath + file, 'r') as f:
			gene = split_file[0]
			length = len(split_file)
			if length == 6:
				op = True
				b = split_file[2]
				gap = split_file[3]
			elif length == 5:
				op = False
				gap = split_file[2]
			else: 
				continue
			seq = split_file[4]
			seqs = []
			contains = []

			for line in f:
				line_seq = line.split(',')[0]
				line_contains = line.split(',')[1]
				if ('Subsequence' in line_seq):
					continue
				seqs.append(line_seq)
				contains.append(int(line_contains.strip()))

			f.close()
			# for i in range(len(seqs)):
			#	 print(seqs[i], contains[i])
			if op:
				entries.append(Entry(gene, seq, seqs, contains, gap, op, b))
			else:
				entries.append(Entry(gene, seq, seqs, contains, gap, op))
	return entries

def reverse_complement(seq):
		return "".join("TGCA"["ACGT".index(s)] for s in seq[::-1])

	# dmd_sequence_r = reverse_complement(dmd_sequence)
'''
#%%
# dna_input must be a constant tensor from tensorflow.constant 
def tf_dna_encode_embedding_table(dna_input, name="dna_encode"):
	"""Map DNA sequence to one-hot encoding using an embedding table."""
	
	# Define the embedding table
	_embedding_values = np.zeros([89, 4], np.float32)
	_embedding_values[ord('G')] = np.array([0, 0, 1, 0])
	_embedding_values[ord('A')] = np.array([1, 0, 0, 0])
 	_embedding_values[ord('C')] = np.array([0, 1, 0, 0])
	_embedding_values[ord('T')] = np.array([0, 0, 0, 1])
	_embedding_values[ord('W')] = np.array([.5, 0, 0, .5])
	_embedding_values[ord('S')] = np.array([0, .5, .5, 0])
	_embedding_values[ord('M')] = np.array([.5, .5, 0, 0])
	_embedding_values[ord('K')] = np.array([0, 0, .5, .5])
	_embedding_values[ord('R')] = np.array([.5, 0, .5, 0])
	_embedding_values[ord('Y')] = np.array([0, .5, 0, .5])
	_embedding_values[ord('B')] = np.array([0, 1. / 3, 1. / 3, 1. / 3])
	_embedding_values[ord('D')] = np.array([1. / 3, 0, 1. / 3, 1. / 3])
	_embedding_values[ord('H')] = np.array([1. / 3, 1. / 3, 0, 1. / 3])
	_embedding_values[ord('V')] = np.array([1. / 3, 1. / 3, 1. / 3, 0])
	_embedding_values[ord('N')] = np.array([.25, .25, .25, .25])

	embedding_table = tf.get_variable(
		'dna_lookup_table', _embedding_values.shape,
		initializer=tf.constant_initializer(_embedding_values),
		trainable=False) # Ensure that embedding table is not trained

	with tf.name_scope(name):
		dna_input = tf.decode_raw(dna_input, tf.uint8) # Interpret string as bytes
	 	dna_32 = tf.cast(dna_input, tf.int32)
		encoded_dna = tf.nn.embedding_lookup(embedding_table, dna_32)
	return encoded_dna
'''

EPCOHS = 100 #	an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
BATCH_SIZE = 32 # a set of N samples. The samples in a batch are processed` independently, in parallel. If training, a batch results in only one update to the model.
OUTPUT_DIM = 64 # Embedding output

DROPOUT_RATIO = 0.2 # proportion of neurones not used for training
MAXSEQ = 26 # cuts text after number of these characters in pad_sequences
RNN_HIDDEN_DIM = 32

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]

def create_lstm(number_of_classes, time_steps, features, metrics = METRICS, rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, dropout = DROPOUT_RATIO, output_bias=None):
	model = Sequential()
	if output_bias is not None:
		output_bias = tf.keras.initializers.Constant(output_bias)
	model.add(Embedding(number_of_classes, OUTPUT_DIM, name='embedding_input_layer', input_length=time_steps, input_shape=(time_steps, features)))
	model.add(TimeDistributed(LSTM(rnn_hidden_dim, return_sequences=True)))
	model.add(Dropout(dropout))
	model.add(TimeDistributed(LSTM(rnn_hidden_dim)))
	model.add(Dropout(dropout))
	model.add(Dense(1, activation='sigmoid', bias_initializer=output_bias))
	model.compile('adam', 'mean_squared_error', metrics=['accuracy'])
	return model

def create_lstm_test(input_dim, input_length, rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, dropout = DROPOUT_RATIO):
	model = Sequential()

	model.add(Embedding(input_dim, OUTPUT_DIM, name='embedding_layer', input_length=input_length))
	model.add(Bidirectional(LSTM(rnn_hidden_dim, return_sequences=True)))
	model.add(Dropout(dropout))
	model.add(Bidirectional(LSTM(rnn_hidden_dim)))
	model.add(Dropout(dropout))
	model.add(Dense(1, activation='sigmoid'))
	model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
	return model

def create_plots(history):
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('accuracy.png')
	plt.clf()

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('loss.png')
	plt.clf()


def letter_to_index(letter):
	_alphabet = 'ATGC'
	return next((i + 1 for i, _letter in enumerate(_alphabet) if _letter == letter), None)


# Connect this df with the create_entries function above
def load_data(entries, test_split = 0.2, MAXSEQ = MAXSEQ):
	matches = 0
	total = 0
	data = pd.DataFrame({
		'seqs': [],
		'contains': []
	})
	max_sentence = 0

	for entry in entries:
		if len(entry.seqs) > max_sentence:
			max_sentence = len(entry.seqs)
	print(max_sentence)
	contain_values = []
	for entry in entries:
		seq_df = pd.DataFrame({
			'col': []
		})
		seq_to_index_df = pd.DataFrame({
			'col': []
		})
		padded_integer_df = pd.DataFrame({
			'col': []
		})

		matches += sum(np.array(entry.contains).astype(int))
		total += len(np.array(entry.contains).astype(int))
		
		seq_df = pd.DataFrame({
			'col': entry.seqs
			})

		seq_to_index_df['col'] = seq_df['col'].apply(lambda x: [int(letter_to_index(e)) for e in x])

		padded_integer_df['col'] = [[0] * (MAXSEQ - len(x)) + x for x in seq_to_index_df['col']] + [[0] * MAXSEQ] * (max_sentence - len(seq_to_index_df['col']))

		if len(entry.contains) < max_sentence:
			entry.contains += [0] * (max_sentence - len(entry.contains))

		contain_values += entry.contains
		# May have to change entry.contains to np.array(entry.contains)
		data = data.append({'seqs': padded_integer_df['col'].tolist(), 'contains': entry.contains}, ignore_index=True)
		
	#print(matches/total)
	from sklearn.model_selection import train_test_split

	train_df, test_df = train_test_split(data, test_size=0.2)
	train_df, val_df = train_test_split(train_df, test_size=0.2)

	train_labels = np.array(train_df['contains'].tolist())
	val_labels = np.array(val_df.pop('contains').tolist())
	test_labels = np.array(test_df.pop('contains').tolist())

	train_features = np.array(train_df['seqs'].tolist())
	val_features = np.array(val_df['seqs'].tolist())
	test_features = np.array(test_df['seqs'].tolist())

	#print(train_labels)

	print('Training labels shape:', train_labels.shape)
	print('Validation labels shape:', val_labels.shape)
	print('Test labels shape:', test_labels.shape)

	print('Training features shape:', train_features.shape)
	print('Validation features shape:', val_features.shape)
	print('Test features shape:', test_features.shape)
	
	return train_features, train_labels, test_features, test_labels, val_features, val_labels


if __name__ == "__main__":
	print ('Gathering entries...')
	entries = create_entries()
	input_dim = len(entries) + 1
	print ('Loading data...')
	train_features, train_labels, test_features, test_labels, val_features, val_labels = load_data(entries)	

	print ('Creating model...')
	model = create_lstm(input_dim, len(train_features[0]), len(train_features[0][0]))
	
	model.summary()
	
	#prediction = model.predict(train_features, verbose = 1, batch_size = 1)
	#results = model.evaluate(train_features, train_labels, batch_size=1, verbose=1)
	#print("Loss: {:0.4f}".format(results[0]))

	print ('Fitting model...')

	history = model.fit(train_features, train_labels, batch_size=4, epochs=EPCOHS, validation_data=(val_features, val_labels), verbose = 1)

	# validate model on unseen data
	score, acc = model.evaluate(train_features, train_labels, batch_size=4, verbose = 1)
	print('Validation score:', score)
	print('Validation accuracy:', acc)

	#create_plots(history)

	quit()
