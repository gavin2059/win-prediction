import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model

## TODO fix timeseries split
## TODO fix reshaping

class  prepData:

        def convert(self, json):
                return pd.read_json(
                        json, orient='records',
                        convert_dates=['time_period_start', 'time_period_end'])

        def prep(self, df, mode):
                # Check for nulls
                if (df.isnull().values.any()):
                        df.dropna(axis = 0, how = 'any', inplace = True)

                # Setup prediction
                y = pd.DataFrame()
                # Output is whether candle moved up
                y['up'] = (df['price_open'] < df['price_close'])
                # Inputs are all available data for now
                features = [ 
                        'price_open', 'price_high', 'price_low', 
                        'price_close', 'volume_traded', 'trades_count'] 
                print(y.head())
                # Scale data for performance and accuracy
                if (mode == 'train'):
                        scaledDf = self.scaleTrain(df, features)
                else:
                        scaledDf = self.scalePredict(df, features)
                # feature variables' values are scaled down to smaller values compared to the real values given above.
                print(scaledDf.head())

                # X_train, y_train, X_test, y_test = self.split(scaledDf, y)

                # # Process data for LSTM
                # trainX = np.array(X_train)
                # testX = np.array(X_test)

                # X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
                # X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

        def scaleTrain(self, df, features):
                self.scaler = MinMaxScaler()
                scaled = self.scaler.fit_transform(df[features])
                return pd.DataFrame(columns = features, data = scaled, index = df.index)

        def scalePredict(self, df, features):
                scaled = self.scaler.transform(df[features])
                return pd.DataFrame(columns = features, data = scaled, index = df.index)

        def split(self, scaledDf, y):
                   # Split into train and test set
                timesplit = TimeSeriesSplit(n_splits = 9) # adv: samples are observed at fixed time intervals
                for train_index, test_index in timesplit.split(scaledDf):
                        X_train, X_test = scaledDf[:len(train_index)], scaledDf[len(train_index): (len(train_index)+len(test_index))]
                        y_train, y_test = y[:len(train_index)].values.ravel(), y[len(train_index): (len(train_index)+len(test_index))].values.ravel()