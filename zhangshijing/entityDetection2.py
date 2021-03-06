import numpy as np
import pandas as pd


print ("Zhang Shijing KBQA entity detection")


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
		if curY[j] == 'I':
			Y[i][j] = True
		else:
			Y[i][j+xMaxLen] = True

#print X,Y
print('X0',X[0])
print('Y0',Y[0])
#for i in range(5):
#	print (Y[i])

#build model
print("build lstm model")
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

model = Sequential()
model.add(Masking(mask_value= 0,input_shape=(xMaxLen,gloveSize)))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(xMaxLen*2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="model/entityDetection2.lstm.model", verbose=1, save_best_only=True,)
history = LossHistory()
result = model.fit(X[:50000], Y[:50000], batch_size=100, nb_epoch=20, verbose=1, validation_data=(X[50001:60000], Y[50001:60000]), callbacks=[checkpointer, history])
model.save('model/entityDetection2.lstm.model')
score = model.evaluate(X[60001:80000], Y[60001:80000], batch_size=100)
print (score)
print(history.losses)
