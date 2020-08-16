import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

validation_set_size = 1500
timeWindow = 4


intermediate_layer_model = load_model('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Models/CNN_intermediate_layer.h5')
intermediate_output_train = intermediate_layer_model.predict([train, robot_state_train_input])
intermediate_output_test = intermediate_layer_model.predict([test, robot_state_test_input])

intermediate_output_train.shape

train_set = intermediate_output_train[0:intermediate_output_train.shape[0]-validation_set_size , : ]
validation_set = intermediate_output_train[intermediate_output_train.shape[0]-validation_set_size: , : ]
test_set = intermediate_output_test[: , :]

train = keras.preprocessing.sequence.TimeseriesGenerator(train_set, train_set, length=timeWindow, sampling_rate=1, stride=1, batch_size=1)
validation = keras.preprocessing.sequence.TimeseriesGenerator(validation_set, validation_set, length=timeWindow, sampling_rate=1, stride=1, batch_size=1)
test = keras.preprocessing.sequence.TimeseriesGenerator(test_set, test_set, length=timeWindow, sampling_rate=1, stride=1, batch_size=1)

train[0][0].shape

print("train_set shape: {}".format(train_set.shape))

train_matrix = np.zeros((train_set.shape[0]-timeWindow, timeWindow, 80))
for i in range(timeWindow,train_set.shape[0]):
  train_matrix[i-timeWindow, : , : ] = train[i-timeWindow][0][0]

validation_matrix = np.zeros((validation_set.shape[0]-timeWindow, timeWindow, 80))
for i in range(timeWindow,validation_set.shape[0]):
  validation_matrix[i-timeWindow, : , : ] = validation[i-timeWindow][0][0]

test_matrix = np.zeros((test_set.shape[0]-timeWindow, timeWindow, 80))
for i in range(timeWindow,test_set.shape[0]):
  test_matrix[i-timeWindow, : , : ] = test[i-timeWindow][0][0]

output_train_matrix = np.zeros((train_set.shape[0]-timeWindow, 7))
for i in range(timeWindow,train_set.shape[0]):
  output_train_matrix[i-timeWindow , :] = robot_state_train_label[i-timeWindow:(i-timeWindow+1)]

output_validation_matrix = np.zeros((validation_set.shape[0]-timeWindow, 7))
for i in range(timeWindow,validation_set.shape[0]):
  output_validation_matrix[i-timeWindow, : ] = robot_state_train_label[(intermediate_output_train.shape[0] - validation_set_size + i - timeWindow):(intermediate_output_train.shape[0] - validation_set_size + i - timeWindow)+1]

output_test_matrix = np.zeros((test_set.shape[0]-timeWindow, 7))
for i in range(timeWindow,test_set.shape[0]):
  output_test_matrix[i - timeWindow,:] = robot_state_test_label[(i - timeWindow):(i - timeWindow)+1]

print("output_test_matrix: {}".format(output_test_matrix.shape))


monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, 
        verbose=1, mode='auto', restore_best_weights=True)


GRU_Model = keras.models.Sequential()
GRU_Model.add(keras.layers.GRU(150, return_sequences=True, input_shape=(timeWindow,80 )))
GRU_Model.add(keras.layers.GRU(100, input_shape=(timeWindow,80 )))
GRU_Model.add(keras.layers.Dense(50))
GRU_Model.add(keras.layers.Dense(40))
GRU_Model.add(keras.layers.Dense(7,activation="linear"))
GRU_Model.compile(loss='mae', optimizer=keras.optimizers.Adam(), metrics=['mse','accuracy'])
GRU_Model.fit(train_matrix ,output_train_matrix, callbacks=[monitor], epochs=40, validation_data=(validation_matrix, output_validation_matrix),verbose=2)
score= GRU_Model.evaluate(test_matrix, output_test_matrix) 



test_predict_gru = GRU_Model.predict(test_matrix)

err_GRU_Model= test_predict_gru - output_test_matrix
Gru_err_mean = np.mean(abs(err_GRU_Model))
a = np.where(err_GRU_Model > 0.01)
a = np.asarray(list(zip(*a)))
print("number of err elements higher than 0.01")
print(a.shape)

GRU_Model.save('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Models/RCN_GRU.h5')

GRU_Model.summary()