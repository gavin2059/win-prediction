import pandas as pd
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
                return scaledDf

        def scaleTrain(self, df, features):
                self.scaler = MinMaxScaler()
                scaled = self.scaler.fit_transform(df[features])
                return pd.DataFrame(columns = features, data = scaled, index = df.index)

        def scalePredict(self, df, features):
                scaled = self.scaler.transform(df[features])
                return pd.DataFrame(columns = features, data = scaled, index = df.index)