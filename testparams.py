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
import time
import keras
import numpy as np
import random 
import sys
from keras import layers
from collections import Counter
import tensorboard 
import pandas as pd

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
    

def create_model(chars,maxlen,hl,lr,pl):
    # hl = hidden layers
    # lr = learning rate
    # pl = parallel layer
    
    
    model=keras.models.Sequential()
    if pl:
        model.add(layers.LSTM(hl,recurrent_dropout=0.3, return_sequences=True, input_shape=(maxlen,len(chars)))) 
        for newl in range( (pl) ):
            model.add(layers.LSTM(hl,recurrent_dropout=0.3))
    else:
        model.add(layers.LSTM(hl,recurrent_dropout=0.5, return_sequences=False, input_shape=(maxlen,len(chars))))
    
    model.add(layers.Dense(len(chars),activation='softmax'))
    optimizer=keras.optimizers.Adam(lr=lr)
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

def train_and_predict(model, train_text, train_x, train_y, no_of_epochs, batch_size, maxlen, char_indices, tblog):
    acc=[]
    val_acc=[]
    loss=[]
    val_loss=[]
    for epoch in range(1,no_of_epochs):
        print('epoch',epoch)

        callbacks_list=[keras.callbacks.ModelCheckpoint(filepath='lstm_char_model.h5')]
        #tensorboardcall = [keras.callbacks.TensorBoard(log_dir=tblog, histogram_freq=1, batch_size=128, 
#                                                       write_graph=True, write_grads=False, 
#                                                       write_images=False, embeddings_freq=0, embeddings_layer_names=None, 
#                                                       embeddings_metadata=None, embeddings_data=None, update_freq='epoch')]

        history=model.fit(train_x,train_y,batch_size=batch_size,epochs=no_of_epochs, validation_split=0.3,
                          callbacks=callbacks_list)

        history_dict=history.history
        loss.append(history_dict['loss'])
        val_loss.append(history_dict['val_loss'])
        acc.append(history_dict['acc'])
        val_acc.append(history_dict['val_acc'])

        start_index=random.randint(0,len(train_text)- maxlen- 1)
        generated_train_text=train_text[start_index:start_index+maxlen]
        expected_output=train_text[start_index+maxlen:start_index+maxlen+400]
        print('---Expected output: ', expected_output)
        print('\n\n---Generating with seed: "'+ generated_train_text + '"')
        
        for temperature in [0.2, 0.5, 1.0]:
            #print('______temperature:', temperature)
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
            #print()
    return acc,loss,val_acc,val_loss
            

loc_train_data=r"simple-examples/data/ptb.char.train.txt"
train_text=load_data(loc_train_data)
#loc_val_data=r"simple-examples/data/ptb.char.valid.txt"
#val_text=load_data(loc_val_data)

#a = train_text.split(' ')

train_text=train_text.replace(' ','');
train_text=train_text.replace('_',' ');

train_text_length=800000
train_text=train_text[0:train_text_length]
print('Corpus Length: ', len(train_text))
#print('Validation Corpus Length: ', len(val_text))



train_counter=Counter(train_text)
#val_counter=Counter(val_text)

maxlen=60
step=3
no_of_epochs=20
batch_size=128

train_sentences, train_label=create_sequences(train_text,maxlen,step)
#val_sentences, val_label=create_sequences(val_text,maxlen,step)




chars, char_indices=mapping(train_text)
train_x, train_y=one_hot_encode(train_sentences, chars, char_indices, train_label, maxlen)
#val_x, val_y=one_hot_encode(val_sentences, chars, char_indices, val_label, maxlen)


for hiddenlayers in [100]:
    for parallellayer in [1]:
        for learningRate in [.01]:
            # already have results for these
            if ((hiddenlayers!=10) or (parallellayer!=0)):
                keras.backend.clear_session()

                start = time.time()
                print("hl: ", hiddenlayers)
                print("parallellayer: ", parallellayer)
                print("learning rate: ", learningRate)
                hl = hiddenlayers
                model=create_model(chars,maxlen,hl,learningRate, parallellayer)

                acc=[]
                loss=[]
                val_acc=[]
                val_loss=[]

                logfile = './logs/hl' + str(hl) + '_lr' + str(learningRate) + '_pl' +str(parallellayer)

                acc, loss,val_acc,val_loss=train_and_predict(model, train_text, train_x, train_y, no_of_epochs,
                                                             batch_size, maxlen, char_indices, logfile)
                allresultsdir = './allresults/'
                allresultsfile = allresultsdir + 'hl' + str(hl) + '_lr' + str(learningRate) + '_pl' +str(parallellayer)

                results = { 'TrainingAcc': acc, "TrainingLoss": loss, "ValidationAcc": val_acc,
                                         "Validation Loss":val_loss}
                df = pd.DataFrame(data=results)
                df.to_csv(allresultsfile, sep=',')



                end = time.time()
                print("total run time", end - start)