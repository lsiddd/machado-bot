#!/usr/bin/env python
# coding: utf-8

from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, SpatialDropout1D
from pickle import dump,load
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.models import load_model

# read a file and return its contents
def read_file(filepath):
	with open(filepath) as f:
		str_text = f.read()
	return str_text

# create model with given parameters
def create_model(vocabulary_size, seq_len, dropout, n_lstms, dense_neurons, lstm_size):

    print('dropout:, ', dropout)
    print('number of lstms:', n_lstms)
    print('number of dense neurons:', dense_neurons)
    print('lstm size:', lstm_size)

    model = Sequential()
    model.add(Embedding(vocabulary_size, seq_len,input_length=seq_len))
    model.add(SpatialDropout1D(dropout))
    for i in range(n_lstms):
        model.add(LSTM(lstm_size, dropout=0.3, recurrent_dropout=dropout))
    model.add(Dense(dense_neurons,activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(vocabulary_size,activation='softmax'))
    opt_adam = optimizers.adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',optimizer=opt_adam,metrics=['accuracy'])
    # model.summary()
    return model

def plot(history):
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



def main():
    # define possible network parameters
    # well sweep through these to find the best archotecture
    dropouts = [0.1, 0.2, 0.3, 0.4]
    lstm_size = [100, 200, 500, 1000]
    n_lstms = [1, 2, 3]
    dense_neurons = [100, 200, 400, 1000]

    # read preprocessed code and split into words
    text = read_file('texts/machado_preprocessed.txt')
    tokens = text.split(" ")

    # create chunks of N words 
    train_len = 10
    text_sequences = []
    for i in range(train_len,len(tokens)):
        seq = tokens[i-train_len:i]
        text_sequences.append(seq)


    sequences = {}
    count = 1
    for i in range(len(tokens)):
        if tokens[i] not in sequences:
            sequences[tokens[i]] = count
            count += 1
            
    # tokenize our vocabulary
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_sequences)
    sequences = tokenizer.texts_to_sequences(text_sequences) 

    # index the vocabulary and collect some statistics
    unique_words = tokenizer.index_word
    unique_wordsApp = tokenizer.word_counts
    vocabulary_size = len(tokenizer.word_counts)

    # add unique sequences to numpy array
    n_sequences = np.empty([len(sequences),train_len], dtype='int32')
    for i in range(len(sequences)):
        n_sequences[i] = sequences[i]


    # take last column of the sequence as target
    train_inputs = n_sequences[:,:-1]
    train_targets = n_sequences[:,-1]

    # make categorical targets
    train_targets = to_categorical(train_targets, num_classes=vocabulary_size+1)
    seq_len = train_inputs.shape[1]
    train_inputs.shape

    dump(tokenizer,open('models/machado_tokenizer_Model4','wb'))


    acc = 0
    history = []
    for drop in dropouts:
        for l_lstm in n_lstms:
            for dense in dense_neurons:
                for lstm_s in lstm_size:
                    model = create_model(vocabulary_size+1,seq_len, drop, l_lstm, dense, lstm_s)
                    path = 'checkpoints/word_pred_Model4.h5'
                    checkpoint = ModelCheckpoint(path, monitor='accuracy', verbose=0, save_best_only=True, mode='min')
                    history = model.fit(train_inputs,train_targets,batch_size=128,epochs=10,verbose=1,
                                        callbacks=[checkpoint], validation_split=0.33)
                    acc_ = history.history['val_accuracy'][-1] # value of lass loss in the model
                    if (acc_ > acc):
                        acc = acc_
                        print(f"accuracy improved to {acc}")
                        model.save('models/machado_modell4.h5')

    plot(history)

if (__name__ == "__main__"):
    main()