import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--modelType', type=str, default = "bilstm", help="lstm, bilstm")
parser.add_argument('--batchSize', type=int, default=32)
parser.add_argument('--yDimension', type=int, default = 2, help="1 or 2")
args = parser.parse_args()
print args.modelType
print args.batchSize
print args.yDimension
modelType = args.modelType
batchSize = args.batchSize
yDimension = args.yDimension

print ("Zhang Shijing KBQA entity detection prediction")


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
allY = allData.iloc[:,7].values


xMaxLen = 0
for i in range(m):
	curLen = len(allX[i].split(" "))
	if curLen > xMaxLen:
		xMaxLen = curLen
print ('language maxLen =', xMaxLen)

X = np.zeros((m, xMaxLen, gloveSize))
Y = np.zeros((m, xMaxLen*2), dtype = np.bool)
if yDimension == 1:
	Y = np.zeros((m, xMaxLen), dtype = np.bool)

for i in range(m):	
	curX = allX[i].lower().split(" ")
	curY = allY[i].split(" ")
	curLen = len(curX)
	for j in range(curLen):
		curWord = curX[j]
		if curWord in gloveMap:
			X[i][j] = gloveMap[curWord]
		else:
			X[i][j] = np.zeros((gloveSize))
		if yDimension == 1:
			if curY[j] == 'I': Y[i][j] = True
		else:
			if curY[j] == 'I':
				Y[i][j] = True
			else:
				Y[i][j+xMaxLen] =True
#print X,Y
print('X0',X[0])
print('Y0',Y[0])
#for i in range(5):
#	print (Y[i])

#load model
print("load lstm model")
import keras.layers
from keras.layers import Masking, Embedding, Bidirectional
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



model = load_model('model/entityDetection' + str(yDimension) + "." + modelType + '.model')

prediction = model.predict(X[75708+10813:])
np.savetxt('prediction/entityDetectionPrediction' + str(yDimension) + "." + modelType + '.predictionData', prediction, fmt='%.6f')
argmaxPrediction = np.argmax(prediction[:,:xMaxLen],axis=1)
#print (prediction)
for i in range(5):
	print prediction[i]
	#print Y[60001+i]
	#print allX[60001+i]

#diy argmax

print("\nDIYargmax")
wrongDIYCnt = 0
wrongArgmaxCnt = 0
wrongContainCnt = 0
wrongSoftContainCnt = 0
for i in range(len(prediction)):
	curLen = len(allX[i+75708+10813].split(" "))
	if yDimension == 2:
		for j in range(xMaxLen):
			if prediction[i][j] > prediction[i][j+xMaxLen] and j < curLen:
				prediction[i][j] = 1;
			else:
				prediction[i][j] = 0;
	else:
		for j in range(xMaxLen):
                        if prediction[i][j] > 0.2 and j < curLen:
                                prediction[i][j] = 1;
                        else:
                                prediction[i][j] = 0;
	
	for j in range(xMaxLen):
		if bool(prediction[i][j]) != Y[i+75708+10813][j]:
			wrongDIYCnt += 1
			'''
			print ("wrong",i, j, bool(prediction[i][j]), Y[i+60001][j])
			print prediction[i]
        		print Y[60001+i]
        		print allX[60001+i]
			'''
			break
	if Y[i+75708+10813][argmaxPrediction[i]] != True:
                wrongArgmaxCnt += 1
	
	for j in range(xMaxLen):
		if Y[i+75708+10813][j] == True and bool(prediction[i][j]) != True:
			wrongContainCnt += 1
			break
	for j in range(xMaxLen):
		if Y[i+75708+10813][j] == True :
			if (j >= 1 and bool(prediction[i][j-1]) == True) or (bool(prediction[i][j]) == True) or (j+1 < xMaxLen and bool(prediction[i][j+1]) == True):
				continue
			else: 
				wrongSoftContainCnt += 1
				break

print("totollySameAccuracy =", (21632-wrongDIYCnt)*1.0/21632)
print("argMaxAccuracy =", (21632-wrongArgmaxCnt)*1.0/21632)
print("containAccuracy =", (21632-wrongContainCnt)*1.0/21632)
print("softContainAccuracy =", (21632-wrongSoftContainCnt)*1.0/21632)	

