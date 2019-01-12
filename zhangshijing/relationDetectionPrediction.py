import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--modelType', type=str, default = "bilstm", help="lstm, bilstm")
parser.add_argument('--batchSize', type=int, default=32)
parser.add_argument('--epoch', type=int, default = 20, help="iter num")
args = parser.parse_args()
print args.modelType
print args.batchSize
print args.epoch
modelType = args.modelType
batchSize = args.batchSize
epoch = args.epoch

print ("Zhang Shijing KBQA relation detection prediction")


#load glove vector
print("load glove vector")
gloveMap ={}
gloveFile = open("miniglove.50d.txt")
line = gloveFile.readline()
while line:
	curLine = line.split(" ")
	curValue = curLine[1:]
	curValue[-1] = curValue[-1][:-1]
	gloveMap[curLine[0]] = curValue
#	print curLine[0],curValue
	line = gloveFile.readline()
    #myDict[curLine[0]] = float(curLine[1])
gloveFile.close()
gloveSize = len(gloveMap['the'])
print ('gloveSize =', gloveSize)

#load train and test data
print("load all data")
allData = pd.read_csv('all.csv', sep = '\t', header = None)
#print (trainData)
'''
testData = pd.read_csv('test.csv', sep = '\t', header = None)
print (testData)
'''
m,n = np.shape(allData)
print (m,n)

allX = allData.iloc[:,6].values
allY = allData.iloc[:,4].values


xMaxLen = 0
relationMap = {}
relationCnt = 0
for i in range(m):
	curLen = len(allX[i].split(" "))
	if curLen > xMaxLen:
		xMaxLen = curLen
	if allY[i] not in relationMap:
		relationMap[allY[i]] = relationCnt
		relationCnt += 1
print ('language maxLen =', xMaxLen)
print ('relationCnt =', relationCnt)

X = np.zeros((m, xMaxLen, gloveSize))
Y = np.zeros((m, relationCnt), dtype = np.bool)

for i in range(m):	
	curX = allX[i].lower().split(" ")
	curLen = len(curX)
	for j in range(curLen):
		curWord = curX[j]
		if curWord in gloveMap:
			X[i][j] = gloveMap[curWord]
		else:
			X[i][j] = np.zeros((gloveSize))
		Y[i][relationMap[allY[i]]] = True
#print X,Y
print('X0',X[0])
print('Y0',Y[0])
#for i in range(5):
#	print (Y[i])

#build model
print("load_model for prediction")
import keras.layers
from keras.layers import Masking, Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []
	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))

model = load_model("model/relationDetection." + modelType + ".model")
prediction = model.predict(X[75708+10813:])
np.savetxt('prediction/relationDetectionPrediction' + "." + modelType + '.predictionData', prediction, fmt='%.6f')
argmaxPrediction = np.argmax(prediction,axis=1)

class node(object):
	def __init__(self,index,value):
		self.index = index
		self.value = value
	index = 0
	value = 0

def topNIndex(inputA, N):
	nodes = []
	for i in range(len(inputA)):
		nodes.append(node(i,inputA[i]))
	nodes.sort(cmp=None,key=lambda x:x.value,reverse=True)
	res = []
	for i in range(N):
		res.append(nodes[i].index)
	return res

topN = 4
argmaxPredictionTop3 = [topNIndex(prediction[i], topN) for i in range(len(prediction))]
for i in range(5):
        print prediction[i]
        print argmaxPrediction[i]
        print argmaxPredictionTop3[i]
	#print Y[10813+i]
        #print allX[60001+i]

wrongArgmaxCnt = 0
wrongTop3ArgmaxCnt = 0
for i in range(len(prediction)):
	if Y[i+75708+10813][argmaxPrediction[i]] != True:
		wrongArgmaxCnt += 1
	#if Y[i+75708+10813][argmaxPredictionTop3[i][0]] != True and Y[i+75708+10813][argmaxPredictionTop3[i][1]] != True and Y[i+75708+10813][argmaxPredictionTop3[i][2]] != True:
	#	wrongTop3ArgmaxCnt += 1
	curSum = 0
	for j in range(topN):
		curSum += Y[i+75708+10813][argmaxPredictionTop3[i][j]]
	if curSum == 0:
		wrongTop3ArgmaxCnt += 1
print("argMaxAccuracy =", (21632-wrongArgmaxCnt)*1.0/21632)
print("top3ContainAccuracy =", (21632-wrongTop3ArgmaxCnt)*1.0/21632)

for i in range(21,50):
	argmaxPredictionTop3 = [topNIndex(prediction[j], i) for j in range(len(prediction))]
	wrongTop3ArgmaxCnt = 0
	for j in range(len(prediction)):
		curSum = 0
		for k in range(i):
			curSum += Y[j+75708+10813][argmaxPredictionTop3[j][k]]
		if curSum == 0:
			wrongTop3ArgmaxCnt += 1
	print("top"+str(i)+"ContainAccuracy =", (21632-wrongTop3ArgmaxCnt)*1.0/21632)
