import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, BatchNormalization

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import math

def letter_to_index(letter):
	_alphabet = 'ATCG'
	return next(((i + 1) / 4 for i, _letter in enumerate(_alphabet) if _letter == letter), None)


def add_buffer(values, n = 2):
	return [e - n / 2 + i for e in values for i in range(n + 1) if e - n / 2 + i > -1]


def score_model(y_act, y_pred, f = 15):
	max_pos_pred = y_pred.argsort()[-f:]
	max_pos_act = y_act.argsort()[-f:]
	agreements = sum([e in max_pos_act for e in max_pos_pred])
	return agreements / f


def score_model_buffer(y_act, y_pred, f = 15):
	max_pos_pred = y_pred.argsort()[-f:]
	max_pos_act = y_act.argsort()[-f:]
	agreements = sum([e in add_buffer(max_pos_act) for e in max_pos_pred])
	return agreements / f


def score_samples(model, X_train, y_train, X_test, y_test):
	y_train_score_list = []
	for i in range(len(X_train)): # len(X_train)
		y_pred = model.predict(X_train[i:i + 1])
		y_train_score_list.append(score_model(y_train[i], y_pred[0]))
	y_test_score_list = []
	for i in range(len(X_test)):
		y_pred = model.predict(X_test[i:i+1])
		y_test_score_list.append(score_model(y_test[i], y_pred[0]))
	return y_train_score_list, y_test_score_list


mypath = 'data/ref_sequences1/'

data = []

k = 15

from os import listdir
from os.path import isfile, join
files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and str(k) in f]

for file in files:
	data.append(pd.read_csv(mypath + file))

for i, entry_data in enumerate(data):
	#print(f'File: {files[i]}')

	query = file.split('_')[0]
	entry_data['kmer'] = entry_data['kmer']

	entry_data['kmer'] = entry_data['kmer'].apply(lambda x: [float(letter_to_index(elem)) for elem in x])

	a = np.array(entry_data['kmer'].tolist())

	a = a.reshape((1, a.shape[0], k * 2))

	b = np.array(entry_data['score'].tolist())

	if 'features' not in locals():
		features = a
		outputs = b
	else:
		features = np.vstack((features, a))
		outputs = np.vstack((outputs, b))


input_shape = features.shape

X_train, X_test, y_train, y_test = train_test_split(features, outputs, test_size = 0.20)

if True:
	model = Sequential()
	model.add(Conv1D(filters=6, kernel_size=16, strides=(1), padding="same", input_shape=input_shape[1:]))
	model.add(BatchNormalization())
	model.add(Dense(16, activation='relu'))
	model.add(MaxPooling1D(pool_size=(2), strides=2))
	model.add(Conv1D(filters=6, kernel_size=16, strides=(1), padding="same", input_shape=input_shape[1:]))
	model.add(BatchNormalization())
	model.add(Dense(32, activation='relu'))
	model.add(MaxPooling1D(pool_size=(2), strides=2))
	model.add(Conv1D(filters=6, kernel_size=16, strides=(1), padding="same", input_shape=input_shape[1:]))
	model.add(BatchNormalization())
	model.add(Dense(64, activation='relu'))
	model.add(MaxPooling1D(pool_size=(2), strides=2))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(input_shape[1:2][0]))
	# Compiles the model
	model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
	model.summary()

if True:
	# fit model
	history = model.fit(X_train, y_train, epochs = 100, batch_size = 80, verbose=1, validation_data=(X_test, y_test))

if True:
	model.load_weights('./checkpoints/my_checkpoint1')
	
if True:
	print('Without buffer')
	y_train_score_list, y_test_score_list = score_samples(model, X_train, y_train, X_test, y_test)

	print('Train average:', sum(y_train_score_list) / len(y_train_score_list))
	print('Test average:', sum(y_test_score_list) / len(y_test_score_list))

	print('With buffer')
	y_train_score_list, y_test_score_list = score_model_buffer(model, X_train, y_train, X_test, y_test)

	print('Train average:', sum(y_train_score_list) / len(y_train_score_list))
	print('Test average:', sum(y_test_score_list) / len(y_test_score_list))

if True:
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model absolute loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('figures/cnn1_' + str(k) + '_abs_loss.png')
	plt.clf()

if True:
	data = pd.DataFrame({'abs_loss': [history.history['loss']], 'abs_val_loss': [history.history['val_loss']]})

	data.to_csv('figures/cnn1_' + str(k) + '.csv')


#print(yhat)
