from polygon import RESTClient
import datetime
import requests
import json

class getData:

    def __init__(self):
        self.key = '6EC069D3-FB83-4BA6-902F-0119CD4F6D17'
        self.headers = {'X-CoinAPI-Key': self.key}

    def getHistorical(self):
        url = ''.join(['https://rest.coinapi.io/v1/', 'ohlcv/BNB/USD/',
        'history?period_id=5MIN', '&time_start=2017-07-24T00:00:00',
        '&time_end=2021-12-23T00:00:00', '&limit=', str(5)])
        return self.get(url)


    def getLastN(self, n):
        url = ''.join(['https://rest.coinapi.io/v1/', 'ohlcv/BNB/USD/',
        'latest?period_id=5MIN', '&limit=', str(n)])
        return self.get(url)


    def get(self, url):
        response = requests.get(url, headers=self.headers)
        response.close()
        if (response.status_code == 200):
            return json.dumps(response.json())
        else:
            raise Exception("Request failed with code " + response.status_code)

    def printRes(self, resp):
        pass

    def exportRes(self, resp, name):
        pass


# a = getData()
# print(a.getLastN(10))
# print(a.getHistorical())