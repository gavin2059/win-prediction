import getData
import prepData
import lstm

# Set up model
getter, prepper = getData.getData(), prepData.prepData()
# testData = getter.getLastN(5)
# histData = getter.getHistorical()
# df = prepper.convert(histData)
fp = open(r"data.txt", "r")
df = prepper.convert(fp.read())
fp.close()
X, y = prepper.prep(df, 'train')
print(X.shape, y.shape)
model = lstm.model(X, y, X.shape, None)
model.createTrainModel()
model.trainNTimes(3)
# model.createPredictModel()