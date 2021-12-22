""" 
TODO: actually use timeseries properly
TODO: adjust data s.t. prediction doesn't have access to current day x
TODO: use actual data from polygon API
"""

# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from keras.layers import LSTM, Dense, Dropout
import matplotlib. dates as mandates
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping

## SETUP TRAINING MODEL

# Early callbacks to prevent overfitting
earlystopping = EarlyStopping(
        monitor='loss',min_delta=0.000000000001,patience=30, restore_best_weights = True)

# Create training model
lstmTrain = Sequential()
lstmTrain.add(LSTM(32, input_shape=(None, trainX.shape[1]), activation='relu', return_sequences=True))
lstmTrain.add(LSTM(1, return_sequences=True))

# Train model
rates = [0.001,0.0001,0.00001]
for rate in rates:
    print('training with lr = ' + str(rate))
    lstmTrain.compile(loss='mse', optimizer=Adam(lr=rate))
    lstmTrain.fit(X_train,y_train,epochs=1000000,callbacks=[earlystopping],verbose=2) #train indefinitely until loss stops decreasing
    print('\n\n\n\n\n')

## SETUP PREDICTION MODEL
lstmPredict = Sequential()
lstmPredict.add(LSTM(32, input_shape=(None, trainX.shape[1]), activation='relu', 
return_sequences=True, stateful=True, batch_input_shape=(1,None,4)))
lstmPredict.add(LSTM(1, return_sequences=False, stateful=True))
lstmPredict.set_weights(lstmTrain.get_weights())
lstmPredict.reset_states()

# Predicted vs True Adj Close Value – LSTM
# plt.plot(y_test, label='True Value')
# plt.plot(y_pred, label='LSTM Value')
# plt.title('Prediction by LSTM')
# plt.xlabel('Time Scale')
# plt.ylabel('Scaled USD')
# plt.legend()
# plt.show()

# Evaluate
# print(lstmTrain.evaluate(X_test, y_test, batch_size = 8))