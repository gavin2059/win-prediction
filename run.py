import getData
import prepData
import lstm
# import lstm

# Set up model
getter, prepper = getData.getData(), prepData.prepData()
# testData = getter.getLastN(5)
# histData = getter.getHistorical()
# df = prepper.convert(histData)
fp = open(r"data.txt", "r")
df = prepper.convert(fp.read())
X, y = prepper.prep(df, 'train')

model = lstm.model(X, y, X.shape, None)