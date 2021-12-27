""" 
TODO: actually use timeseries properly
TODO: adjust data s.t. prediction doesn't have access to current day x
"""

# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout, Flatten, Bidirectional
import matplotlib. dates as mandates
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
from keras.models import load_model
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

class model:

    def __init__(self, X, y, trainShape, batchShape):
        self.X, self.y = X, y
        self.trainShape = trainShape
        self.batchShape = batchShape
    
    def createTrainModel(self):
        # Create training model
        self.lstmTrain = Sequential()
        self.lstmTrain.add(
            Bidirectional(
                LSTM(
                    units=128, input_shape=(self.trainShape[1], self.trainShape[2]), 
                    activation='relu', return_sequences=True, stateful=False)
                )
            )
        self.lstmTrain.add(Dropout(rate=0.2))
        self.lstmTrain.add(Dense(1, activation='sigmoid'))
        self.lstmTrain.compile(loss='binary_crossentropy', 
        optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    def trainNTimes(self, n):
        # Split into train and test set
        timesplit = TimeSeriesSplit(n_splits = n) # adv: samples are observed at fixed time intervals
        i, score = 0, []
        for tr_index, val_index in timesplit.split(self.X):
            X_tr, X_val = self.X[tr_index], self.X[val_index]
            y_tr, y_val = self.y[tr_index], self.y[val_index]
            self.trainOnce(X_tr, y_tr)
            i += 1
            print(self.lstmTrain.evaluate(X_val, y_val))
            if (i == n):
                print(self.lstmTrain.summary())
                # print(self.lstmTrain.predict(X_val)[-1])
                # print(self.lstmTrain.evaluate(X_val, y_val))

    def trainOnce(self, X_tr, y_tr):
        # # Early callbacks to prevent overfitting
        # earlystopping = EarlyStopping(
        #         monitor='loss',min_delta=0.000000000001,patience=30, restore_best_weights = True)

        # Train model
        print('training with lr = ' + str(0.0001))
        # self.lstmTrain.fit(
        #     X_tr,y_tr,epochs=20,
        #     callbacks=[earlystopping], validation_split=2.0/9.0,
        #     verbose=2) #train indefinitely until loss stops decreasing
        self.lstmTrain.fit(X_tr,y_tr,epochs=20,verbose=2)
        print('\n\n\n\n\n')

    def createPredictModel(self):
        # Create prediction model based on training model results
        self.lstmPredict = Sequential()
        self.lstmPredict.add(LSTM(32, input_shape=self.trainShape, activation='relu', 
        return_sequences=True, stateful=True, batch_input_shape=self.batchShape))
        self.lstmTrain.add(Flatten())
        self.lstmPredict.add(Dense(1))
        self.lstmPredict.set_weights(self.lstmTrain.get_weights())
        self.lstmPredict.reset_states()

    def updatePredictModel(self):
        self.lstmPredict.set_weights(self.lstmTrain.get_weights())

    def predict(self, X):
        return self.lstmPredict.predict(X)

## SETUP PREDICTION MODEL


# Predicted vs True Adj Close Value – LSTM
# plt.plot(y_test, label='True Value')
# plt.plot(y_pred, label='LSTM Value')
# plt.title('Prediction by LSTM')
# plt.xlabel('Time Scale')
# plt.ylabel('Scaled USD')
# plt.legend()
# plt.show()

# Evaluate
# print(self.lstmTrain.evaluate(X_test, y_test, batch_size = 8))