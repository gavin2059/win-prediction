import getData
import prepData
import lstm

# Set up model
getter, prepper = getData(), prepData()
histData = getter.getHistorical()
model = lstm(prepper.prep(histData))
model.train()