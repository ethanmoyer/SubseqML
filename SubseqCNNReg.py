import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, BatchNormalization

import tensorflow.keras.backend as K 

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import math

def letter_to_index(letter):
	_alphabet = 'ATCG'
	return next(((i + 1) / 4 for i, _letter in enumerate(_alphabet) if _letter == letter), None)


def score_model(y_act, y_pred, f = 15):
	max_pos_act_list = []
	v = K.variable(0.)
	for i in range(f):
		max_pos_act = K.argmax(y_act)[-f:]
		max_pos_pred = K.argmax(y_pred)[-f:]

	agreements = sum([e in max_pos_act for e in max_pos_pred])
	print(agreements / f)
	return agreements / f


mypath = 'data/ref_sequences2/'

data = []

k = 15

from os import listdir
from os.path import isfile, join
files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and str(k) in f]

for file in files[:5]:
	data.append(pd.read_csv(mypath + file))

for i, entry_data in enumerate(data):
	#print(f'File: {files[i]}')

	query = file.split('_')[0]
	entry_data['kmer'] = entry_data['kmer'] + query

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
	history = model.fit(X_train, y_train, epochs = 5, batch_size = 80, verbose=1, validation_data=(X_test, y_test))


if False:
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model absolute loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('figures/cnn3_' + str(k) + '_abs_loss.png')
	plt.clf()

if False:
	a = [math.sqrt(e) for e in history.history['loss']]
	plt.plot(a)
	a = [math.sqrt(e) for e in history.history['val_loss']]
	plt.plot(a)
	plt.title('model relative loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('figures/cnn3_' + str(k) + '_rel_loss.png')
	plt.clf()

if False:
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('figures/cnn3_' + str(k) + '_accuracy.png')
	plt.clf()



#print(yhat)
