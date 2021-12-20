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
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib. dates as mandates
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping

# Visualize data
df = pd.read_csv("BNB.csv", na_values = ['null'],
index_col = 'Date', parse_dates = True, infer_datetime_format = True)
df.head()

# Check for nulls
if (df.isnull().values.any()):
        df.dropna(axis = 0, how = 'any', inplace = True)

# Plot the True Adjusted Close Value (closing value of stock on day)
# df['Adj Close'].plot()

# Setup prediction
y = pd.DataFrame(df['Adj Close']) # Set dependent var
features = ['Open', 'High', 'Low', 'Volume'] # Set indep vars

# Scale data for performance and accuracy
scaler = MinMaxScaler()
featureTransform = scaler.fit_transform(df[features])
featureTransform = pd.DataFrame(columns = features, data = featureTransform, index = df.index)
# feature variables' values are scaled down to smaller values compared to the real values given above.
print(featureTransform.head())

# Split into train and test set
timesplit = TimeSeriesSplit(n_splits = 9) # adv: samples are observed at fixed time intervals
for train_index, test_index in timesplit.split(featureTransform):
        X_train, X_test = featureTransform[:len(train_index)], featureTransform[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = y[:len(train_index)].values.ravel(), y[len(train_index): (len(train_index)+len(test_index))].values.ravel()

# Process data for LSTM
# num = len(trainX) * 2.0/9.0
trainX = np.array(X_train)
testX = np.array(X_test)

X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

# Early callbacks to prevent overfitting
earlystopping = EarlyStopping(
        monitor ="val_loss", mode ="min", patience = 5, restore_best_weights = True)

# Create model
lstm = Sequential()
lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
# plot_model(lstm, show_shapes=True, show_layer_names=True)


# Train model
history=lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False,
                validation_split = 2.0/9.0, callbacks = [earlystopping])


# LSTM Prediction
y_pred= lstm.predict(X_test)

# Predicted vs True Adj Close Value – LSTM
plt.plot(y_test, label='True Value')
plt.plot(y_pred, label='LSTM Value')
plt.title('Prediction by LSTM')
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.legend()
plt.show()

# Evaluate
print(lstm.evaluate(X_test, y_test, batch_size = 8))