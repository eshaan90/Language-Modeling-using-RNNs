# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 04:14:02 2018

@author: evkirpal
"""

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense,Dropout
from keras.preprocessing.text import Tokenizer
from keras import callbacks,optimizers,backend
from keras.models import Sequential
import keras.utils as ku
import numpy as np
import random
import matplotlib.pyplot as plt



def dataset_preparation(data,maxlen,step):

    tokenizer.fit_on_texts([data])
    total_words = len(tokenizer.word_index) + 1
    encoded = tokenizer.texts_to_sequences([data])[0]
        
    # determine the vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    
    #Create n-gram sequences
    sequences = list()
    for i in range(0, len(encoded)-maxlen,step):
        j=i+1
        while j<i+maxlen:
            seq=encoded[i:j+1]
            sequences.append(seq)
            j=j+2
            
##    	sequence = encoded[i:i+maxlen]
##    	sequences.append(sequence)

    max_sequence_len = max([len(x) for x in sequences])
    input_sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))
    print('Total Sequences: %d' % len(input_sequences))

    
    # split into X and y elements
##    sequences = np.array(sequences)
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=vocab_size)
    return predictors, label, max_sequence_len, total_words


def create_model(predictors, label, max_sequence_len, total_words):
    model = Sequential()
    model.add(Embedding(total_words, 500, input_length=max_sequence_len-1))
    model.add(LSTM(400 , return_sequences = True))
    model.add(Dropout(0.5))
    model.add(LSTM(400))
    model.add(Dropout(0.5))
    model.add(Dense(total_words, activation='softmax'))
    optimizer=optimizers.Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model 



def generate_text(model, seed_text, next_words, max_sequence_len, no_of_epochs):
	
    acc=[]
    val_acc=[]
    loss=[]
    val_loss=[]
    for epoch in range(1,no_of_epochs):
        print('epoch {} of {}'.format(epoch,no_of_epochs-1))
        
        callbacks_list=[callbacks.ModelCheckpoint(filepath='lstm_word_model.h5')]
        history=model.fit(predictors, label, epochs=no_of_epochs, batch_size=128, validation_split=0.2,verbose=1,callbacks=callbacks_list)
        
        history_dict=history.history
        loss.append(history_dict['loss'][0])
        val_loss.append(history_dict['val_loss'][0])
        acc.append(history_dict['acc'][0])
        val_acc.append(history_dict['val_acc'][0])
    
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
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

backend.clear_session()

file2 = open(r"simple-examples/data/ptb.train.txt","r") 
data=file2.read()
data=data[0:4000000]
data=data.replace('-',' - ');


maxlen=40
step=3
no_of_epochs=10
#batch_size=128

next_words=40

#data = open('data.txt').read()


tokenizer = Tokenizer()
predictors, label, max_sequence_len, total_words= dataset_preparation(data,maxlen,step)
model = create_model(predictors, label, max_sequence_len, total_words)


data_list=data.split()
start_index=random.randint(0,len(data_list)- max_sequence_len- 1)
seed_text=' '.join(data_list[start_index:start_index + max_sequence_len])
print('---Generating with seed: "'+ seed_text + '"')

loss, val_loss, acc, val_acc=generate_text(model,seed_text, next_words, max_sequence_len, no_of_epochs)

print('Trianing Loss= {}'.format(loss))
print('Testing Loss= {}'.format(val_loss))
print('Trianing Acc= {}'.format(acc))
print('Trianing Acc= {}'.format(val_acc))

plt.figure(1)
plt.plot(range(no_of_epochs-1),loss,)
plt.plot(range(no_of_epochs-1),val_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.legend(labels=['Training','Validation'],loc='best')
plt.grid()
plt.show()


plt.figure(2)
plt.plot(range(no_of_epochs-1),acc,)
plt.plot(range(no_of_epochs-1),val_acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.legend(labels=['Training','Validation'],loc='best')
plt.grid()
plt.show()