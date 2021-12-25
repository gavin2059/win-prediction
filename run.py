import getData
import prepData
import json
# import lstm

# Set up model
getter, prepper = getData.getData(), prepData.prepData()
testData = getter.getLastN(5)
df = prepper.convert(testData)
prepper.prep(df)
# histData = getter.getHistorical()
# model = lstm(prepper.prep(histData))
# model.train()