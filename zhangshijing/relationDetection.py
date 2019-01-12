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

print ("Zhang Shijing KBQA relation detection")


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
print("build lstm model")
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

model = Sequential()
model.add(Masking(mask_value= 0,input_shape=(xMaxLen,gloveSize)))
model.add(Bidirectional(LSTM(128, dropout_W=0.2, dropout_U=0.2)))
model.add(Dense(relationCnt, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="model/relationDetection." + modelType + ".model", verbose=1, save_best_only=True,)
history = LossHistory()
result = model.fit(X[:75708], Y[:75708], batch_size=batchSize, nb_epoch=epoch, verbose=1, validation_data=(X[75708:75708+10813], Y[75708:75708+10813]), callbacks=[checkpointer, history])
model.save("model/relationDetection." + modelType + ".model")
score = model.evaluate(X[75708+10813:], Y[75708+10813:], batch_size=batchSize)
print (score)
print(history.losses)
