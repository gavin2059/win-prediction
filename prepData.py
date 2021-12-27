import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

class  prepData:

        def convert(self, json):
                return pd.read_json(
                        json, orient='records',
                        convert_dates=['time_period_start', 'time_period_end'])

        def prep(self, df, mode):
                # Check for nulls
                # if (df.isnull().values.any()):
                #         df.dropna(axis = 0, how = 'any', inplace = True)

                # Setup prediction
                y = pd.DataFrame()
                # Output is whether candle moved up / down
                y['up'] = (df['price_open'] < df['price_close']) * 1
                # y['down'] = (df['price_open'] > df['price_close']) * 1
                # y['same'] = (df['price_open'] == df['price_close']) * 1
                # y['up'] = pd.Series([1 for x in range(100000)])
                # y['down'] = pd.Series([0 for x in range(100000)])
                # y['same'] = pd.Series([0 for x in range(100000)])
                print(y.head())
                # Inputs are all available data for now
                features = [ 
                        'price_open', 'price_high', 'price_low', 
                        'price_close', 'volume_traded', 'trades_count'] 

                # Scale data for performance and accuracy
                if (mode == 'train'):
                        scaledDf = self.scaleTrain(df, features)
                else:
                        scaledDf = self.scalePredict(df, features)
                # feature variables' values are scaled down to smaller values compared to the real values given above.
                return self.reshape(scaledDf, y, 100, 1)

        def scaleTrain(self, df, features):
                self.scaler = MinMaxScaler()
                scaled = self.scaler.fit_transform(df[features])
                return pd.DataFrame(columns = features, data = scaled, index = df.index)

        def scalePredict(self, df, features):
                scaled = self.scaler.transform(df[features])
                return pd.DataFrame(columns = features, data = scaled, index = df.index)

        def reshape(self, scaledDf, y, lookback, lookahead):
                x = np.reshape(scaledDf.to_numpy(), ((100000 // lookback), lookback, 6))
                x = x[:100000 // lookback - lookback]
                y = np.reshape(y.to_numpy(), (100000 // lookback, lookback, 1))
                y = y[lookback + lookahead - 1: ]
                return x, y

                