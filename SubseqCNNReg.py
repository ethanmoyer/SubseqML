import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, BatchNormalization


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics

import matplotlib.pyplot as plt
import pickle
import math
from random import seed

from os import listdir
from os.path import isfile, join

nuc_list = ['A', 'T', 'C', 'G']
nuc_series = pd.Series(nuc_list)
nuc_encoder = np.array(pd.get_dummies(nuc_series))

def one_hot_encode_sequence(seq):
	seq_ = []
	for e in seq:
		seq_ += nuc_encoder[nuc_list.index(e)].tolist()
	return np.array(seq_)


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


def score_samples_buffer(model, X_train, y_train, X_test, y_test):
	y_train_score_list = []
	for i in range(len(X_train)): # len(X_train)
		y_pred = model.predict(X_train[i:i + 1])
		y_train_score_list.append(score_model_buffer(y_train[i], y_pred[0]))
	y_test_score_list = []
	for i in range(len(X_test)):
		y_pred = model.predict(X_test[i:i+1])
		y_test_score_list.append(score_model_buffer(y_test[i], y_pred[0]))
	return y_train_score_list, y_test_score_list


def create_reg_cnn_model(input_shape):
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
	return model

if False:
	mypath = 'data/ref_sequences0/'

	data = []

	k = 15

	files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and str(k) in f]

	for file in files:
		data.append(pd.read_csv(mypath + file))

	for i, entry in enumerate(data):
		print(f'File: {files[i]}')

		# Look into this
		if len(entry) != 985:
			continue

		query = file.split('_')[0]

		entry['kmer'] = entry['kmer'].apply(lambda x: one_hot_encode_sequence(x))

		a = np.array(entry['kmer'].tolist())

		a = a.reshape((1, a.shape[0], k * len(nuc_list)))

		b = np.array(entry['score'].tolist())

		if 'features' not in locals():
			features = a
			outputs = b
		else:
			features = np.vstack((features, a))
			outputs = np.vstack((outputs, b))

	with open("data/ref_seq0_data", "wb") as fp: pickle.dump((features, outputs), fp, protocol=4)

if False:
	quit()

(features, outputs) = pickle.load(open("data/ref_seq0_data", 'rb'))

input_shape = features.shape

# Cross-Validate
# https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_05_2_kfold.ipynb

seed(42)

kf = KFold(10, shuffle=True, random_state=42) # Use for KFold classification

early_stopping = EarlyStopping(patience=5)

# Out of sample
oos_y = []
oos_pred = []

fold = 0

for train, test in kf.split(features):
	fold += 1
	print(f"Fold #{fold}")

	x_train = features[train]
	y_train = outputs[train]
	x_test = features[test]
	y_test = outputs[test]

	model = create_reg_cnn_model(input_shape)
	history = model.fit(x_train, y_train, epochs = 100, batch_size = 80, verbose=1, validation_data=(x_test, y_test), callbacks=[early_stopping])

	pred = model.predict(x_test)
	oos_y.append(y_test)
	oos_pred.append(pred)

	# Measure this fold's RMSE
	rmse = np.sqrt(metrics.mean_squared_error(pred, y_test))
	print(f"Fold RMSE: {rmse}")

	# Measure the accuracy of the model at each fold.
	score_without_buffer = 0
	for i in range(len(pred)):
		score_without_buffer += score_model(pred[i], y_test[i])
	print(f'Avg fold score without a buffer: {score_without_buffer / len(pred)}')

	score_with_buffer = 0
	for i in range(len(pred)):
		score_with_buffer += score_model_buffer(pred[i], y_test[i])
	print(f'Avg fold score without a buffer: {score_with_buffer / len(pred)}')


# (oos_y, oos_pred) = pickle.load(open("data/ref_seq0_crossval", 'rb'))

# This is all out of sample

oos_y = np.concatenate(oos_y)
oos_pred = np.concatenate(oos_pred)
rmse = metrics.mean_squared_error(oos_pred,oos_y)
print(f"Final, out of sample score (RMSE): {rmse}")

score_without_buffer = 0
for i in range(len(oos_y)):
	score_without_buffer += score_model(oos_pred[i], oos_y[i])

print(f'Avg fold score without a buffer: {score_without_buffer / len(oos_y)}')

score_with_buffer = 0
for i in range(len(oos_y)):
	score_with_buffer += score_model_buffer(oos_pred[i], oos_y[i])

print(f'Avg fold score without a buffer: {score_with_buffer / len(oos_y)}')

if False:
	train_loss = []
	val_loss = []
	train_score = []
	val_score = []
	train_score_buffer = []
	val_score_buffer = []

if False:
	for q in range(10):
		print('Percentage complete: ', round(q / 100 * 100, 2), '%', sep='')
		history = model.fit(X_train, y_train, epochs = 100, batch_size = 80, verbose=0, validation_data=(X_test, y_test), callbacks=[early_stopping])
		train_loss.append(history.history['loss'][99])
		val_loss.append(history.history['val_loss'][99])
		y_train_score_list, y_test_score_list = score_samples(model, X_train, y_train, X_test, y_test)
		train_score.append(sum(y_train_score_list) / len(y_train_score_list))
		val_score.append(sum(y_test_score_list) / len(y_test_score_list))
		y_train_score_list, y_test_score_list = score_samples_buffer(model, X_train, y_train, X_test, y_test)
		train_score_buffer.append(sum(y_train_score_list) / len(y_train_score_list))
		val_score_buffer.append(sum(y_test_score_list) / len(y_test_score_list))
	with open("data/ref_sequences0_data/train_loss0", "wb") as fp: pickle.dump(train_loss, fp)
	with open("data/ref_sequences0_data/val_loss0", "wb") as fp: pickle.dump(val_loss, fp)
	with open("data/ref_sequences0_data/train_score0", "wb") as fp: pickle.dump(train_score, fp)
	with open("data/ref_sequences0_data/val_score0", "wb") as fp: pickle.dump(val_score, fp)
	with open("data/ref_sequences0_data/train_score_buffer0", "wb") as fp: pickle.dump(train_score_buffer, fp)
	with open("data/ref_sequences0_data/val_score_buffer0", "wb") as fp: pickle.dump(val_score_buffer, fp)

if False:
	model.save_weights('./checkpoints/my_checkpoint0')

if False:
	model.load_weights('./checkpoints/my_checkpoint1')
	
if False:
	print('Without buffer')
	y_train_score_list, y_test_score_list = score_samples(model, X_train, y_train, X_test, y_test)
	print('Train average:', sum(y_train_score_list) / len(y_train_score_list))
	print('Test average:', sum(y_test_score_list) / len(y_test_score_list))

if False:
	print('With buffer')
	y_train_score_list, y_test_score_list = score_samples_buffer(model, X_train, y_train, X_test, y_test)
	print('Train average:', sum(y_train_score_list) / len(y_train_score_list))
	print('Test average:', sum(y_test_score_list) / len(y_test_score_list))

if False:
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model absolute loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('figures/cnn0_' + str(k) + '_abs_loss.png')
	plt.clf()

if False:
	data = pd.DataFrame({'abs_loss': [history.history['loss']], 'abs_val_loss': [history.history['val_loss']]})

	data.to_csv('figures/cnn0_' + str(k) + '.csv')


#print(yhat)
