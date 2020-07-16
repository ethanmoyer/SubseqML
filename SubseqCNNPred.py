import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, BatchNormalization

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import math

def letter_to_index(letter):
	_alphabet = 'atgc'
	return next(((i + 1) / 4 for i, _letter in enumerate(_alphabet) if _letter == letter), None)

mypath = '/Users/ethanmoyer/Desktop/Coursework/1920_spring/VIP/subsequence-matching/data/ref_sequences/'

data = []

from os import listdir
from os.path import isfile, join
files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and '20' in f]

for file in files:
	data.append(pd.read_csv(mypath + file))

for i, entry_data in enumerate(data[:1000]):
	print(f'File: {files[i]} - {i / 1000 * 100}')
	query = file.split('_')[1]
	entry_data['Subsequence'] = entry_data['Subsequence'] + query

	entry_data['Subsequence'] = entry_data['Subsequence'].apply(lambda x: [float(letter_to_index(elem)) for elem in x])

	a = np.array(entry_data['Subsequence'].tolist())

	a = a.reshape((1, a.shape[0], 40))

	b = np.array(entry_data['Contains'].tolist())

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
	model.add(Conv1D(3, 8, strides=(1), padding="same", input_shape=input_shape[1:]))
	model.add(BatchNormalization())
	model.add(Dense(8, activation='relu'))
	model.add(MaxPooling1D(pool_size=(2), strides=2))

	model.add(Conv1D(3, 16, strides=(1), padding="same", input_shape=input_shape[1:]))
	model.add(BatchNormalization())
	model.add(Dense(16, activation='relu'))
	model.add(MaxPooling1D(pool_size=(2), strides=2))

	model.add(Conv1D(3, 32, strides=(1), padding="same", input_shape=input_shape[1:]))
	model.add(BatchNormalization())
	model.add(Dense(32, activation='relu'))
	model.add(MaxPooling1D(pool_size=(2), strides=2))

	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(981))

	# Compiles the model
	model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse'])

	model.summary()

if True:

	# fit model
	history = model.fit(X_train, y_train, epochs = 50, batch_size = 16, verbose=1, validation_data=(X_test, y_test))

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model absolute loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('cnn_data/cnn0_20_1000_abs_loss.png')
	plt.clf()

	a = [math.sqrt(e) for e in history.history['loss']]
	plt.plot(a / np.mean(y_train))
	a = [math.sqrt(e) for e in history.history['val_loss']]
	plt.plot(a / np.mean(y_test))
	plt.title('model relative loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')

	plt.savefig('cnn_data/cnn0_20_1000_abs_loss.png')
#print(yhat)