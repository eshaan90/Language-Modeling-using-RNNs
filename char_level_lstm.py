# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 10:36:10 2018

@author: evkirpal
"""

#Dependencies
from keras import models,layers,optimizers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import argparse
from keras.preprocessing.text import Tokenizer


# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()


file2 = open(r"simple-examples/data/ptb.char.train.txt","r") 
raw_text=file2.read()


tokenizer = Tokenizer()
tokenizer.fit_on_texts([raw_text])
encoded = tokenizer.texts_to_sequences([raw_text])[0]

# load text
raw_text = load_doc('rhyme.txt')
print(raw_text)

# clean
tokens = raw_text.split()
raw_text = ' '.join(tokens)

# organize into sequences of characters
length = 10
sequences = list()
for i in range(length, len(raw_text)):
	# select sequence of tokens
	seq = raw_text[i-length:i+1]
	# store
	sequences.append(seq)
print('Total Sequences: %d' % len(sequences))

# save sequences to file
out_filename = 'char_sequences.txt'
save_doc(sequences, out_filename)


# load
in_filename = 'char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')

chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))

sequences = list()
for line in lines:
	# integer encode line
	encoded_seq = [mapping[char] for char in line]
	# store
	sequences.append(encoded_seq)
    
# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]

sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X =np.array(sequences)
y = to_categorical(y, num_classes=vocab_size)

# define model
optim=optimizers.Adam(lr=0.001)
model = models.Sequential()
model.add(layers.LSTM(100, input_shape=(X.shape[1], X.shape[2])))
model.add(layers.Dense(vocab_size, activation='softmax'))
print(model.summary())

# compile model
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
# fit model
history=model.fit(X, y, batch_size=30, epochs=5, verbose=2)
print(history)
model.save('char_lstm_model.h5')


# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
	in_text = seed_text
	# generate a fixed number of characters
	for _ in range(n_chars):
		# encode the characters as integers
		encoded = [mapping[char] for char in in_text]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# one hot encode
		encoded = to_categorical(encoded, num_classes=len(mapping))
		# predict character
		yhat = model.predict_classes(encoded, verbose=0)
		# reverse map integer to character
		out_char = ''
		for char, index in mapping.items():
			if index == yhat:
				out_char = char
				break
		# append to input
		in_text += char
	return in_text

# load the model
model = load_model('model.h5')
# load the mapping
mapping = load(open('mapping.pkl', 'rb'))
in_text='Sing a son'
seq_length=10
n_chars=20

# test start of rhyme
print(generate_seq(model, mapping, 10, 'Sing a son', 20))
# test mid-line
print(generate_seq(model, mapping, 10, 'king was i', 20))
# test not in original
print(generate_seq(model, mapping, 10, 'hello worl', 20))