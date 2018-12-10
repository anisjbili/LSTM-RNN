# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:43:55 2018

@author: ASUS
"""
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
#dataset = numpy.loadtxt("pima-indians-diabetes.csv")
dataset = pd.read_csv('spam.csv' ,encoding = "ISO-8859-1")
labels = dataset['v1']
Text = dataset['v2']
tk = Tokenizer(nb_words=2000)
tk.fit_on_texts(Text)
Text = tk.texts_to_sequences(Text)

labels = labels.map({'spam':0, 'ham':1})
Text_Train, Text_Test, labels_train, labels_test = train_test_split(Text,labels,test_size=0.33)

Text_Train = sequence.pad_sequences(Text_Train, 150)
Text_Test = sequence.pad_sequences(Text_Test, 150)

batch_size =20
epochs = 10
num_classes = 15

model = Sequential() 
model.add(Embedding(2000, 50, input_length=150)) 
model.add(Dropout(0.2)) 
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2)) 
model.add(Dense(250, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
train = model.fit(Text_Train, labels_train, batch_size=batch_size,epochs=epochs)#,validation_data=(valid_X, valid_label))
eva = model.evaluate(Text_Test, labels_test, verbose=0)


print('Test loss:', eva[0])
print('Test accuracy:', eva[1])