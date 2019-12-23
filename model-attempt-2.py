#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, SpatialDropout1D
from pickle import dump,load
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.models import load_model


# In[2]:


def read_file(filepath):
	with open(filepath) as f:
		str_text = f.read()
	return str_text


# In[3]:



text = read_file('./machado_preprocessed.txt')
tokens = text.split(" ")
#  tokens.pop(0)


# In[4]:


train_len = 10
text_sequences = []
for i in range(train_len,len(tokens)):
    seq = tokens[i-train_len:i]
    text_sequences.append(seq)


# In[5]:


sequences = {}
count = 1
for i in range(len(tokens)):
    if tokens[i] not in sequences:
        sequences[tokens[i]] = count
        count += 1
        


# In[6]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences) 

#Collecting some information   
unique_words = tokenizer.index_word
unique_wordsApp = tokenizer.word_counts
vocabulary_size = len(tokenizer.word_counts)


# In[7]:


n_sequences = np.empty([len(sequences),train_len], dtype='int32')
for i in range(len(sequences)):
    n_sequences[i] = sequences[i]


# In[8]:


train_inputs = n_sequences[:,:-1]
train_targets = n_sequences[:,-1]

train_targets = to_categorical(train_targets, num_classes=vocabulary_size+1)
seq_len = train_inputs.shape[1]
train_inputs.shape


# In[9]:


def create_model(vocabulary_size, seq_len):
    model = Sequential()
    model.add(Embedding(vocabulary_size, seq_len,input_length=seq_len))
    model.add(SpatialDropout1D(0.3))
    model.add(LSTM(256, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(1000,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(vocabulary_size,activation='softmax'))
    opt_adam = optimizers.adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',optimizer=opt_adam,metrics=['accuracy'])
    model.summary()
    return model


# In[10]:


dump(tokenizer,open('machado_tokenizer_Model4','wb'))


# In[11]:


model = create_model(vocabulary_size+1,seq_len)
path = './checkpoints/word_pred_Model4.h5'
checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='min')
history = model.fit(train_inputs,train_targets,batch_size=128,epochs=40,verbose=1,callbacks=[checkpoint], validation_split=0.33)
model.save('machado_modell4.h5')


# In[12]:


from matplotlib import pyplot

# history = model.fit(X, Y, epochs=100, validation_data=(valX, valY))
pyplot.style.use("classic")
pyplot.grid()
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.savefig("validation_loss_dropout.pdf")

