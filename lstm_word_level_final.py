# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 04:14:02 2018

@author: evkirpal
"""

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense,Dropout
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras import callbacks
from keras.models import Sequential
import keras.utils as ku
from keras.utils import to_categorical
import numpy as np
import random


data = """The cat and her kittens
They put on their mittens,
To eat a Christmas pie.
The poor little kittens
They lost their mittens,
And then they began to cry.
O mother dear, we sadly fear
We cannot go to-day,
For we have lost our mittens."
"If it be so, ye shall not go,
For ye are naughty kittens."""



tokenizer = Tokenizer()

def dataset_preparation(data):

	# basic cleanup
#	corpus = data.lower().split("\n")
#
#	# tokenization	
#	tokenizer.fit_on_texts(corpus)
#	total_words = len(tokenizer.word_index) + 1
#    l=text_to_word_sequence(data,lower=True,split=' ')
#
#	# create input sequences using list of tokens
#    train_sentences, train_label=create_sequences(train_text,maxlen,step)
#    
#	input_sequences = []
#	for line in corpus:
#		token_list = tokenizer.texts_to_sequences([line])[0]
#		for i in range(1, len(token_list)):
#			n_gram_sequence = token_list[:i+1]
#			input_sequences.append(n_gram_sequence)
#            
#
#	# pad sequences 
#	max_sequence_len = max([len(x) for x in input_sequences])
#	input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
#
#	# create predictors and label
#	predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
#	label = ku.to_categorical(label, num_classes=total_words)




    tokenizer.fit_on_texts([data])
    total_words = len(tokenizer.word_index) + 1
    encoded = tokenizer.texts_to_sequences([data])[0]
        
    # determine the vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    
    sequences = list()
    maxlen=40
    step=3
    for i in range(0, len(encoded)-maxlen,step):
    	sequence = encoded[i:i+maxlen]
    	sequences.append(sequence)
    print('Total Sequences: %d' % len(sequences))

    
    # split into X and y elements
    sequences = np.array(sequences)
    predictors, label = sequences[:,:-1],sequences[:,-1]
    label = ku.to_categorical(label, num_classes=vocab_size)
    return predictors, label, maxlen, total_words


def generate_text(seed_text, next_words, max_sequence_len, no_of_epochs):
	
    acc=[]
    val_acc=[]
    loss=[]
    val_loss=[]
    for epoch in range(1,no_of_epochs):
        print('epoch {} of {}'.format(epoch,no_of_epochs-1))
        
        callbacks_list=[callbacks.ModelCheckpoint(filepath='lstm_word_model.h5')]
        history=model.fit(predictors, label, epochs=1, batch_size=128, validation_split=0.2,verbose=1,callbacks=callbacks_list)
        
        history_dict=history.history
        loss.append(history_dict['loss'])
        val_loss.append(history_dict['val_loss'])
        acc.append(history_dict['acc'])
        val_acc.append(history_dict['val_acc'])
    
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            predicted = model.predict_classes(token_list, verbose=0)
        		
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word
    return seed_text


def create_model(predictors, label, max_sequence_len, total_words):
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=maxlen-1))
    model.add(LSTM(150, return_sequences = True))
    model.add(Dropout(0.5))
    model.add(LSTM(150))
    model.add(Dropout(0.5))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model 


file2 = open(r"simple-examples/data/ptb.train.txt","r") 
data=file2.read()
data=data[0:600000]
data=data.replace('-',' - ');

#train_text=data.split()

maxlen=40
step=3
no_of_epochs=3
batch_size=128

next_words=80

#data = open('data.txt').read()

predictors, label, maxlen, total_words= dataset_preparation(data)
model = create_model(predictors, label, maxlen, total_words)

start_index=random.randint(0,len(data)- maxlen- 1)
seed_text=data[start_index:start_index + maxlen]
print('---Generating with seed: "'+ seed_text + '"')

print(generate_text(seed_text, 3, maxlen))


print(generate_text("we naughty", 40, maxlen))