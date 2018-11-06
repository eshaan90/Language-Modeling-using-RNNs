# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 04:14:02 2018

@author: evkirpal
"""

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense,Dropout
from keras.preprocessing.text import Tokenizer
from keras import callbacks,optimizers
from keras.models import Sequential
import keras.utils as ku
import numpy as np
import random
import matplotlib.pyplot as plt



def dataset_preparation(data):

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


def create_model(predictors, label, max_sequence_len, total_words):
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=maxlen-1))
    model.add(LSTM(50, return_sequences = True))
    model.add(Dropout(0.5))
    model.add(LSTM(50))
    model.add(Dropout(0.5))
    model.add(Dense(total_words, activation='softmax'))
    optimizer=optimizers.Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model 


def sample(preds,temperature=1.0):
    preds=np.asarray(preds).astype('float64')
    preds=np.log(preds)/temperature
    exp_preds=np.exp(preds)
    preds=exp_preds/np.sum(exp_preds)
    probas=np.random.multinomial(1,preds,1)
    return np.argmax(probas)


def generate_text(model, seed_text, next_words, maxlen, no_of_epochs):
	
    acc=[]
    val_acc=[]
    loss=[]
    val_loss=[]
    for epoch in range(1,no_of_epochs):
        print('epoch {} of {}'.format(epoch,no_of_epochs-1))
        
        callbacks_list=[callbacks.ModelCheckpoint(filepath='lstm_word_model.h5')]
        history=model.fit(predictors, label, epochs=1, batch_size=128, validation_split=0.2,verbose=1,callbacks=callbacks_list)
        
        history_dict=history.history
        loss.append(history_dict['loss'][0])
        val_loss.append(history_dict['val_loss'][0])
        acc.append(history_dict['acc'][0])
        val_acc.append(history_dict['val_acc'][0])
    
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=maxlen-1, padding='pre')
            predicted = model.predict_classes(token_list, verbose=0)
#        	next_index = sample(preds, temperature)
#            next_word = words[next_index]	
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word
        print(seed_text)
    return loss, val_loss, acc, val_acc



file2 = open(r"simple-examples/data/ptb.train.txt","r") 
data=file2.read()
data=data[0:1500000]
data=data.replace('-',' - ');

#train_text=data.split()

maxlen=40
step=3
no_of_epochs=10
#batch_size=128

next_words=40

#data = open('data.txt').read()


tokenizer = Tokenizer()
predictors, label, maxlen, total_words= dataset_preparation(data)
model = create_model(predictors, label, maxlen, total_words)


data_list=data.split()
start_index=random.randint(0,len(data_list)- maxlen- 1)
seed_text=' '.join(data_list[start_index:start_index + maxlen])
print('---Generating with seed: "'+ seed_text + '"')

loss, val_loss, acc, val_acc=generate_text(model,seed_text, next_words, maxlen, no_of_epochs)

plt.figure(1)
plt.plt(range(no_of_epochs),loss)
plt.plot(range(no_of_epochs),val_loss)
plt.legend()
plt.show()
#print(generate_text("we naughty", 40, maxlen))