#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 01:34:57 2018

@author: MyReservoir
"""

"""
Created on Sun Nov  4 17:39:08 2018
@author: evkirpal
"""

import keras
import numpy as np
import random 
import sys
from keras import layers
from collections import Counter

def load_data(data):
    #Loading data
    file = open(data,"r") 
    text=file.read().lower()
    return text
    
def mapping(train_text):
    #Mapping b/w char and integers
    chars=sorted(list(set(train_text)))
    print('Unique Characters:',len(chars))    
    char_indices = dict((char, chars.index(char)) for char in chars)
    return chars,char_indices

def one_hot_encode(sentences, chars, char_indices, next_chars, maxlen):
    #One-hot Encoding the input and output
    print('Vectorization...')
    x=np.zeros((len(sentences),maxlen,len(chars)),dtype=np.bool)
    y=np.zeros((len(sentences),len(chars)),dtype=np.bool)

    for i,sentence in enumerate(sentences):
        for t,char in enumerate(sentence):
            x[i,t,char_indices[char]]=1
        if next_chars[i]!='*':
            y[i,char_indices[next_chars[i]]]=1
        else:
            y[i,char_indices['_']]=1
    return x,y


def create_sequences(text,maxlen,step):
    #Create sequences and list of label words
    sentences=[]
    next_chars=[]
    for i in range(0,len(text)-maxlen-4,step):
        sentences.append(text[i:i+maxlen])
        next_chars.append(text[i+maxlen])
    print('Number of sequences: ',len(sentences))
    return sentences,next_chars
    

def create_model(chars,maxlen):
    model=keras.models.Sequential()
    model.add(layers.LSTM(500,recurrent_dropout=0.5, return_sequences=True, input_shape=(maxlen,len(chars))))
    model.add(layers.LSTM(500,recurrent_dropout=0.5))
    model.add(layers.Dense(len(chars),activation='softmax'))
    optimizer=keras.optimizers.Adam(lr=0.01)
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

def train_and_predict(model, train_text, train_x, train_y, val_x, val_y, no_of_epochs, batch_size, maxlen, char_indices):
    acc=[]
    val_acc=[]
    loss=[]
    val_loss=[]
    for epoch in range(1,no_of_epochs):
        print('epoch',epoch)
        
        callbacks_list=[keras.callbacks.ModelCheckpoint(filepath='lstm_char_model.h5')]
        history=model.fit(train_x,train_y,batch_size=batch_size,epochs=1, validation_data=(val_x,val_y),callbacks=callbacks_list)
        
        history_dict=history.history
        loss.append(history_dict['loss'])
        val_loss.append(history_dict['val_loss'])
        acc.append(history_dict['acc'])
        val_acc.append(history_dict['val_acc'])
        
        start_index=random.randint(0,len(train_text)- maxlen- 1)
        generated_train_text=train_text[start_index:start_index+maxlen]
        print('---Generating with seed: "'+ generated_train_text + '"')
        
        for temperature in [0.2,0.5,1.0,1.2]:
            print('______temperature:', temperature)
            sys.stdout.write(generated_train_text)
            # We generate 400 characters
            for i in range(400):
                sampled = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(generated_train_text):
                    sampled[0, t, char_indices[char]] = 1.
    
                preds = model.predict(sampled, verbose=0)[0]
                next_index = sample(preds, temperature)
                next_char = chars[next_index]
    
                generated_train_text += next_char
                generated_train_text = generated_train_text[1:]
    
                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()
    return acc,loss,val_acc,val_loss
            
                  
            
loc_train_data=r"simple-examples/data/ptb.char.train.txt"
train_text=load_data(loc_train_data)
loc_val_data=r"simple-examples/data/ptb.char.valid.txt"
val_text=load_data(loc_val_data)

#a = train_text.split(' ')

train_text_length=1000000
train_text=train_text[0:train_text_length]
print('Corpus Length: ', len(train_text))
print('Validation Corpus Length: ', len(val_text))

val_text=val_text.replace('*','_');

train_counter=Counter(train_text)
val_counter=Counter(val_text)

maxlen=60
step=3
no_of_epochs=20
batch_size=128
sentences=[]
next_chars=[]
train_sentences, train_label=create_sequences(train_text,maxlen,step)
val_sentences, val_label=create_sequences(val_text,maxlen,step)



chars, char_indices=mapping(train_text)
train_x, train_y=one_hot_encode(train_sentences, chars, char_indices, train_label, maxlen)
val_x, val_y=one_hot_encode(val_sentences, chars, char_indices, val_label, maxlen)

model=create_model(chars,maxlen)

acc=[]
loss=[]
val_acc=[]
val_loss=[]
acc, loss,val_acc,val_loss=train_and_predict(model, train_text, train_x, train_y, val_x, val_y, 
                                             no_of_epochs, batch_size, maxlen, char_indices)

print("Training Acc: ", acc)
print("Training Loss: ", loss)
print("Validation Acc: ", val_acc)
print("Validation Loss: ", val_loss)