# -*- coding: utf-8 -*-
"""
Created on Sat May  2 22:15:51 2020

@author: Ahmed_Abohgeazy
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


def model(X_train):
  #creating a sequential model
  model = Sequential()
  #adding an LSTM hidden layer with and input shape will vary depending on the number of time steps that we want to memorize
  model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
  #adding an extra LSTM hidden layer
  model.add(LSTM(units = 50, return_sequences = True))
  # Adding the output layer
  model.add(Dense(units = 1))

  return model

def model_4layers(X_train):
  #creating a sequential model
  model = Sequential()
  #adding an LSTM hidden layer with and input shape will vary depending on the number of time steps that we want to memorize
  model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
  #adding an extra LSTM hidden layer
  model.add(LSTM(units = 50, return_sequences = True))
  #adding an extra LSTM hidden layer
  model.add(LSTM(units = 50, return_sequences = True))
  #adding an extra LSTM hidden layer
  model.add(LSTM(units = 50))
  # Adding the output layer
  model.add(Dense(units = 1))

  return model


def model_4layers_dropout(X_train):
  #creating a sequential model
  model = Sequential()
  # Adding the first LSTM layer and some Dropout regularisation
  model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
  #adding a dropout with rate 0.2 to prevent overfitting
  model.add(Dropout(0.2))
  #adding an extra LSTM hidden layer
  model.add(LSTM(units = 50, return_sequences = True))
  #adding a dropout
  model.add(Dropout(0.2))
  #adding an extra LSTM hidden layer
  model.add(LSTM(units = 50, return_sequences = True))
  #adding a dropout
  model.add(Dropout(0.2))
  #adding an extra LSTM hidden layer
  model.add(LSTM(units = 50))
  #adding a dropout
  model.add(Dropout(0.2))
  # Adding the output layer
  model.add(Dense(units = 1))

  return model

# Importing the training set with 5 years
data_train = pd.read_csv('Google_Stock_Price_Train.csv')
#selecting the open stock price
train = data_train.iloc[:, 1:2].values


# 20 finacial days existing in January 2017
data_test = pd.read_csv('Google_Stock_Price_Test.csv')
test = data_test.iloc[:, 1:2].values


# Feature Scaling using normalization
sc = MinMaxScaler(feature_range = (0, 1))
train_scaled = sc.fit_transform(train)

# Creating a data structure with X timesteps and 1 output it's expecting the 
#input after being feature scaled using normalization.
def dataStructure_timesteps(train_scaled, steps):
  X_train = []
  y_train = []
  for i in range(steps, 1258):
      X_train.append(train_scaled[i-steps:i, 0])
      y_train.append(train_scaled[i, 0])
  X_train, y_train = np.array(X_train), np.array(y_train)

  # Reshaping
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

  return X_train, y_train


X_train, y_train = dataStructure_timesteps(train_scaled,60)
model = model_4layers_dropout(X_train)

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mae'])
early_stopping=keras.callbacks.EarlyStopping(patience=10)
model_checkpoint=keras.callbacks.ModelCheckpoint("bestmodel.h5",save_best_only=True)

model.fit(X_train, y_train, epochs = 100, batch_size = 32 ,callbacks=[early_stopping,model_checkpoint])

def prepareTest(steps,data_train,data_test):
  # Getting the predicted stock price of 2017
  dataset_total = pd.concat((data_train['Open'], data_test['Open']), axis = 0)
  # to get the first day of 2017 -steps to get the lower bounds of the dataset, . values wil convert into
  # numpy array
  inputs = dataset_total[len(dataset_total) - len(data_test) - steps:].values
  #
  inputs = inputs.reshape(-1,1)
  #feature scaling applying the same
  inputs = sc.transform(inputs)
  X_test = []

  for i in range(steps, steps+20):
      X_test.append(inputs[i-steps:i, 0])
  X_test = np.array(X_test)
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

  return X_test

X_test = prepareTest(60,data_train,data_test)
predicted_stock_price = model.predict(X_test)
#inversing the scaled values to the real values
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(test, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.savefig('plot.png')
plt.show()

    
