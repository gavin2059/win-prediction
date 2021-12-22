from polygon import RESTClient
import datetime

class getData:

    def __init__(self):
        self.key = "LfhlSKO9dfVHLLNp9SuHpqTKd5yVpEt2"
        self.client = RESTClient(self.key)
        self.ticker = 'X:BNBUSD'
        self.multiplier, self.timespan = 5, 'minute'
        self.dateFrom, self.dateTo = "2017-07-23", "2019-12-24"

    def getHistorical(self):
        dateFrom, dateTo = "2017-07-23", "2019-12-24"
        # dateTo = str(datetime.date.today())
        resp = self.client.crypto_aggregates(
            self.ticker, self.multiplier, self.timespan, 
            self.dateFrom, self.dateTo, limit=50000)
        self.printRes(resp)

    def getLastN(self, n):
        resp = self.client.crypto_aggregates(
            self.ticker, self.multiplier, self.timespan, 
            self.dateFrom, self.dateTo, limit=n, sort='desc')
        self.printRes(resp)

    def printRes(self, resp):
        print(resp.ticker, resp.adjusted, resp.resultsCount)
        for result in resp.results:
            print(result['o'], result['c'], 
            datetime.datetime.fromtimestamp(result['t']/1000.0))

a = getData()
a.getHistorical()