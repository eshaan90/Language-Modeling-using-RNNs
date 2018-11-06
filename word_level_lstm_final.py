# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 23:38:04 2018

@author: evkirpal
"""


import time
import keras
import numpy as np
import random 
import sys
from keras import layers
from collections import Counter
import tensorboard 
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


def mapping(train_text):
    #Mapping b/w word and integers
    words=sorted(list(set(train_text)))
    print('Unique wordacters:',len(words))    
    word_indices = dict((word, words.index(word)) for word in words)
    return words,word_indices


def one_hot_encode(sentences, words, word_indices, next_words, maxlen):
    #One-hot Encoding the input and output
    print('Vectorization...')
    x=np.zeros((len(sentences),maxlen,len(words)),dtype=np.bool)
    y=np.zeros((len(sentences),len(words)),dtype=np.bool)

    for i,sentence in enumerate(sentences):
        for t,word in enumerate(sentence):
            x[i,t,word_indices[word]]=1
            y[i,word_indices[next_words[i]]]=1
    return x,y


def create_sequences(text,maxlen,step):
    #Create sequences and list of label words
    sentences=[]
    next_word=[]
    for i in range(0,len(text)-maxlen,step):
        string=''
        for j in range(i,i+maxlen):
            string+=train_text[j]+' '
        string=string[:-1]
        string=string.split()
        sentences.append(string)
        next_word.append(text[i+maxlen])
    print('Number of sequences: ',len(sentences))
    return sentences,next_word
    

def create_model(words,maxlen,vocab_size):
    model=keras.models.Sequential()
    model.add(layers.Embedding(vocab_size, 10, input_length=maxlen))
    model.add(layers.LSTM(200,recurrent_dropout=0.5, return_sequences=True, input_shape=(maxlen,len(words))))
    model.add(layers.LSTM(200,recurrent_dropout=0.5))
    model.add(layers.Dense(len(words),activation='softmax'))
    optimizer=keras.optimizers.Adam(lr=1)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc'])
    model.summary()
    return model

def sample(preds,temperature=1.0):
    preds=np.asarray(preds).astype('float64')
    preds=np.log(preds)/temperature
    exp_preds=np.exp(preds)
    preds=exp_preds/np.sum(exp_preds)
    probas=np.random.multinomial(1,preds,1)
    return np.argmax(probas)

def train_and_predict(model, train_text, train_x, train_y, no_of_epochs, batch_size, maxlen, word_indices):
    acc=[]
    val_acc=[]
    loss=[]
    val_loss=[]
    for epoch in range(1,no_of_epochs):
        print('epoch {} of {}'.format(epoch,no_of_epochs))
        
        callbacks_list=[keras.callbacks.ModelCheckpoint(filepath='lstm_word_model.h5')]
        history=model.fit(train_x,train_y,batch_size=batch_size,epochs=1, validation_split=0.2,callbacks=callbacks_list)
        
        history_dict=history.history
        loss.append(history_dict['loss'])
        val_loss.append(history_dict['val_loss'])
        acc.append(history_dict['acc'])
        val_acc.append(history_dict['val_acc'])
        
        start_index=random.randint(0,len(train_text)- maxlen- 1)
        generated_text_list=train_text[start_index:start_index+maxlen]
        generated_train_text=' '.join(generated_text_list)
        print('---Generating with seed: "'+ generated_train_text + '"')
        
        for temperature in [0.2,0.5,1.0,1.2]:
            print('______temperature:', temperature)
            sys.stdout.write(generated_train_text)
            # We generate 400 wordacters
            for i in range(40):
                sampled = np.zeros((1, maxlen, len(words)))
                for t, word in enumerate(generated_text_list):
                    sampled[0, t, word_indices[word]] = 1.
    
                preds = model.predict(sampled, verbose=0)[0]
                next_index = sample(preds, temperature)
                next_word = words[next_index]
    
                generated_train_text = generated_train_text+ ' '+ next_word
                generated_train_text = generated_train_text[1:]
    
                sys.stdout.write(next_word)
                sys.stdout.flush()
            print()
    return acc,loss,val_acc,val_loss



file2 = open(r"simple-examples/data/ptb.train.txt","r") 
data=file2.read()
data=data[0:700000]
data=data.replace('-',' - ');

train_text=data.split()

maxlen=40
step=3
no_of_epochs=1
batch_size=128

#
## integer encode text
#tokenizer = Tokenizer()
#tokenizer.fit_on_texts([data])
#encoded = tokenizer.texts_to_sequences([data])[0]



train_sentences, train_label=create_sequences(train_text,maxlen,step)
#val_sentences, val_label=create_sequences(val_text,maxlen,step)


# determine the vocabulary size
words, word_indices=mapping(train_text)
train_x, train_y=one_hot_encode(train_sentences, words, word_indices, train_label, maxlen)

vocab_size = word_indeices + 1
print('Vocabulary Size: %d' % vocab_size)

model=create_model(words,maxlen,vocab_size)

acc=[]
loss=[]
val_acc=[]
val_loss=[]
acc, loss, val_acc, val_loss=train_and_predict(model, train_text, train_x, train_y, no_of_epochs, batch_size, maxlen, word_indices)

print("Training Acc: ", acc)
print("Training Loss: ", loss)
print("Validation Acc: ", val_acc)
print("Validation Loss: ", val_loss)


