import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model

class  PrepData:

        def __init__(self):
                # Visualize data
                self.df = pd.read_csv("BNB.csv", na_values = ['null'],
                index_col = 'Date', parse_dates = True, infer_datetime_format = True)
                self.df.head()

                # Check for nulls
                if (self.df.isnull().values.any()):
                        self.df.dropna(axis = 0, how = 'any', inplace = True)

        def prep(self):
                # Setup prediction
                y = pd.DataFrame(self.df['Adj Close']) # Set dependent var
                features = ['Open', 'High', 'Low', 'Volume'] # Set indep vars

                # Scale data for performance and accuracy
                scaler = MinMaxScaler()
                featureTransform = scaler.fit_transform(self.df[features])
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