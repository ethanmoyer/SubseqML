import tensorflow as tf
# import theano
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Activation, Lambda, Bidirectional, TimeDistributed, Flatten
# from tensorflow.keras.engine import Input, Model, InputSpec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
# from tensorflow.keras.utils import plot_model
# from tensorflow.keras.utils.data_utils import get_file
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
# from tensorflow.keras import backend as K
# from tensorflow.keras.preprocessing import sequence
# from tensorflow.keras.models import model_from_json
import os
# import pydot
# import graphviz

mypath = 'data2/'

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
	from os import listdir
	from os.path import isfile, join
	files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	entries = []
	for file in files[:5]:
		split_file = file.split('_')
		print(split_file)
		with open('data2/' + file, 'r') as f:
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


EPCOHS = 100 #  an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
BATCH_SIZE = 500 # a set of N samples. The samples in a batch are processed` independently, in parallel. If training, a batch results in only one update to the model.
OUTPUT_DIM = 64 # Embedding output

DROPOUT_RATIO = 0.2 # proportion of neurones not used for training
MAXSEQ = 26 # cuts text after number of these characters in pad_sequences
RNN_HIDDEN_DIM = 64

def create_lstm(number_of_classes, time_steps, features, rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, dropout = DROPOUT_RATIO):
	model = Sequential()

	model.add(Embedding(number_of_classes, OUTPUT_DIM, name='embedding_input_layer', input_length=features, input_shape=(time_steps, features)))

	model.add(TimeDistributed(LSTM(rnn_hidden_dim, return_sequences=True)))
	model.add(Dropout(dropout))
	model.add(TimeDistributed(LSTM(rnn_hidden_dim)))
	model.add(Dropout(dropout))
	model.add(Dense(input_dim, activation='sigmoid'))
	model.add(LSTM(rnn_hidden_dim, input_shape=(time_steps, features), return_sequences=True))
	model.add(TimeDistributed(Dense(1)))

	model.compile('adam', 'mean_squared_error', metrics=['accuracy'])
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
def load_data(entries, test_split = 0.4, MAXSEQ = MAXSEQ):
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
		letter_to_str_index_df = pd.DataFrame({
			'col': []
		})
		padded_integer_df = pd.DataFrame({
			'col': []
		})

		matches += sum(np.array(entry.contains).astype(int))
		total += len(np.array(entry.contains).astype(int))
		
		letter_to_str_index_df['col'] = pd.DataFrame({
			'col': entry.seqs
			})['col'].apply(lambda x: [int(letter_to_index(e)) for e in x])

		padded_integer_df['col'] = [[0] * (MAXSEQ - len(x)) + x for x in letter_to_str_index_df['col']] + [[0] * MAXSEQ] * (max_sentence - len(letter_to_str_index_df['col']))
		temp_seqqs = padded_integer_df['col']

		if len(entry.contains) < max_sentence:
			entry.contains += [0] * (max_sentence - len(entry.contains))
		contain_values += entry.contains
		data = data.append({'seqs': temp_seqqs, 'contains': np.array(entry.contains)}, ignore_index=True)
		
	#print(matches/total)

	train_size = int(len(data) * (1 - test_split))
	X_train = data['seqs'][:train_size]
	y_train = data['contains'].values[:train_size]
	X_train = np.array(X_train.tolist()).astype(np.int64)
	y_train = y_train

	y_data = np.array(contain_values)
	y_data = y_data.reshape(len(data), max_sentence, 1)
	y_data_train = y_data[:train_size]
	print(y_data_train)
	print(y_train[0])

	print('X_train')
	print(type(X_train))
	print(X_train.shape)
	print(X_train)

	print('X_train[0]')
	print(type(X_train[0]))
	print(X_train[0])

	print('X_train[0][0]')
	print(type(X_train[0][0]))
	print(X_train[0][0])

	print('y_train')
	print(len(y_train))
	print(type(y_train))
	print(y_train)

	print('y_train[0]')
	print(len(y_train[0]))
	print(type(y_train[0]))
	print(y_train[0])

	print('y_train[0][0]')
	#print(len(y_train[0][0]))
	print(type(y_train[0][0]))
	print(y_train[0][0])

	print('X_train.shape')
	print(X_train.shape)

	X_test = data['seqs'][train_size:]
	y_test = data['contains'][train_size:]
	X_test = np.array(X_test.tolist())
	y_test = np.array(y_test.tolist())

	return X_train, y_data_train, X_train, y_test


if __name__ == "__main__":
	print ('Gathering entries...')
	entries = create_entries()
	input_dim = len(entries) + 1
	print ('Loading data...')
	X_train, y_train, X_test, y_test = load_data(entries)	

	print ('Creating model...')
	model = create_lstm(input_dim, len(X_train[0]), len(X_train[0][0]))

	model.summary()
	
	print ('Fitting model...')	
	length = 5
	seq = np.array([i/float(length) for i in range(2 * length)])
	X = seq.reshape(2, length, 1)
	y = seq.reshape(2, length, 1)
	print(X)
	print(y)
	print(type(y))
	print(type(y[0]))
	print(type(y[0][0]))

	history = model.fit(X_train, y_train, batch_size=20, epochs=EPCOHS, validation_data=(X_train, y_train), verbose = 1)
	quit()
	# validate model on unseen data
	score, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
	print('Validation score:', score)
	print('Validation accuracy:', acc)

	quit()
