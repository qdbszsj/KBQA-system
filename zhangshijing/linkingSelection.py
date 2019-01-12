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

print ("Zhang Shijing KBQA entity linking")

features = pd.DataFrame({
	"entityFindRange":[0]*21632,
	"languageLength":[0]*21632,
	"trueEntityLength":[0]*21632,
	"useFulTop1Entity":[0]*21632,
	"useFulTop1Relation":[0]*21632,
	"numOfCandidateEntity":[0]*21632,
	"numOfEntityLinkings":[0]*21632,
	"topWitchEntitySelected":[0]*21632,
	"topWitchRelationSelected":[0]*21632,	
	"findTrueEntity":[0]*21632,
	"findTrueRelation":[0]*21632,
	"findAns":[0]*21632,
	"correct":[0]*21632
})

print features

#load train and test data
print("load all data")
allData = pd.read_csv('all.csv', sep = '\t', header = None)
#print (trainData)
m,n = np.shape(allData)
print (m,n)

allX = allData.iloc[:,6].values
allEntityY = allData.iloc[:,7].values
allRelationY = allData.iloc[:,4].values
allEntityYName = allData.iloc[:,2].values


relationMap = {}
relationIndexMap = {}
relationCnt = 0
xMaxLen = 0
for i in range(m):
        curLen = len(allX[i].split(" "))
	if i>=75708+10813: 
		features["languageLength"][i-75708-10813] = curLen
		features["trueEntityLength"][i-75708-10813] = len(allEntityYName[i].split(" "))
        if curLen > xMaxLen:
                xMaxLen = curLen
        if allRelationY[i][3:] not in relationMap:
		#print allRelationY[i][3:]
                relationMap[allRelationY[i][3:]] = relationCnt
		relationIndexMap[relationCnt] = allRelationY[i][3:]
                relationCnt += 1
print ('relationCnt =', relationCnt)
print ('language maxLen =', xMaxLen)

X = [[" " for i in range(xMaxLen)] for j in range(m)]
entityY = np.zeros((m, xMaxLen*2), dtype = np.bool)
relationY = np.zeros((m, relationCnt), dtype =np.bool)
print(np.shape(X))


for i in range(m):	
	curX = allX[i].lower().split(" ")
	curEntityY = allEntityY[i].split(" ")
	curLen = len(curX)
	for j in range(curLen):
		X[i][j] = curX[j]
		
	if curEntityY[j] == 'I':
		entityY[i][j] = True
	else:
		entityY[i][j+xMaxLen] =True
	relationY[i][relationMap[allRelationY[i][3:]]] = True
	if i%10000 == 0:
		print ("preprocessing" + str(i) + "/" +str(m))

print "loading entityPrediction"
entityPrediction = pd.read_csv('prediction/entityDetectionPrediction2.bilstm.predictionData', sep = ' ', header = None).astype('float').values
print "loading relationPrediction"
relationPrediction =  pd.read_csv("prediction/relationDetectionPrediction.bilstm.predictionData", sep = ' ', header = None).astype('float').values
print "loading entityNames"
entityNames = pd.read_csv('FB5M.en-name.txt', sep = '\t', header = None).astype('str').values
entityNameMap = {}
for i in range(len(entityNames)):
	if entityNames[i][2].lower() not in entityNameMap:
		entityNameMap[entityNames[i][2].lower()] = [entityNames[i][0][6:-1]]
	else:
		entityNameMap[entityNames[i][2].lower()].append(entityNames[i][0][6:-1])

entityArgmaxPrediction = np.argmax(entityPrediction[:,:xMaxLen],axis=1)

entityCandidates = []

for i in range(len(entityPrediction)):
        curLen = len(allX[i+75708+10813].split(" "))
        for j in range(curLen):
        	if entityPrediction[i][j] > entityPrediction[i][j+xMaxLen] and j < curLen:
                	entityPrediction[i][j] = 1;
                else:
                        entityPrediction[i][j] = 0;
	curEntityArgmaxPrediction = np.argmax(entityPrediction[i][:curLen])
	l = curEntityArgmaxPrediction
	r = curEntityArgmaxPrediction
	while l-1 >= 0 and entityPrediction[i][l-1] == 1:
		l -= 1
	while r < curLen and entityPrediction[i][r] == 1:
		r += 1
	curEntityWords = X[i+75708+10813][l]
	for j in range(l+1, r): 
		curEntityWords = curEntityWords + " " + X[i+75708+10813][j]
	curEntityCandidates = []
	if curEntityWords in entityNameMap:
		curEntityCandidates.append(curEntityWords)
	else:
		features["entityFindRange"][i] = 1
	if l>0: l -= 1
	if r<curLen: r += 1
	for j in range(r-l,0,-1):#length
		for k in range(l,r-j+1):#start position
			curEntityWords = X[i+75708+10813][k]
			for p in range(k+1, k+j):
				curEntityWords = curEntityWords + " " + X[i+75708+10813][p]
			if curEntityWords in entityNameMap:
				if len(curEntityCandidates)>=1 and curEntityWords == curEntityCandidates[0]:
					continue
				curEntityCandidates.append(curEntityWords)		
	if len(curEntityCandidates) == 0:#cannot find candidates
		features["entityFindRange"][i] = 2
		l = 0
		r = curLen
		for j in range(r-l,0,-1):#length	
			for k in range(l,r-j+1):#start position
				curEntityWords = X[i+75708+10813][k]
				for p in range(k+1, k+j):
					curEntityWords = curEntityWords + " " + X[i+75708+10813][p]
				if curEntityWords in entityNameMap:
					if len(curEntityCandidates)>=1 and curEntityWords == curEntityCandidates[0]:
						continue
					curEntityCandidates.append(curEntityWords)
	entityCandidates.append(curEntityCandidates)
	features["numOfCandidateEntity"][i] = len(curEntityCandidates)
	if i%10000 == 0:
		print ("entityCandidates " + str(i) + "/100000")

print np.shape(entityCandidates)

for i in range(10):
	print entityCandidates[i]

print "loading entity-relation map"

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

entityRelations = pd.read_csv('freebase-FB5M.txt', sep = '\t', header = None).astype('str').values
print entityRelations

entityRelationsSet = set()
for i in range(len(entityRelations)):
	entityRelationsSet.add((entityRelations[i][0][19:], entityRelations[i][1][17:].replace("/", ".")))
	if i%1000000 == 0:
		print "entityRelationsSet " + str(i) + "/12000000"

ans = []
correctCnt = 0
topN = 4

#for Daisy
output = open("prediction/tureAnswerabilityForDaisy.tsv", "w")

for i in range(len(entityCandidates)):
	argmaxPredictionTopN = topNIndex(relationPrediction[i], topN) 
	findAns = 0
	correctAns = 0
	for j in range(len(entityCandidates[i])):
		for k in range(topN):
			curRelation = relationIndexMap[argmaxPredictionTopN[k]]	
			entityList = entityNameMap[entityCandidates[i][j]]
			for p in range(len(entityList)):
				curEntity = entityList[p]
				if (curEntity,curRelation) in entityRelationsSet:
					ans.append([curEntity,curRelation])
					#print entityCandidates[i][j], allEntityYName[i+75708+10813].lower(), curRelation, allRelationY[i+75708+10813][3:]
					if j == 0:
						features["useFulTop1Entity"][i] = 1
					if k == 0:
						features["useFulTop1Relation"][i] = 1
					features["numOfEntityLinkings"][i] = len(entityList)
					features["topWitchEntitySelected"][i] = j
					features["topWitchRelationSelected"][i] = k
					if entityCandidates[i][j] == allEntityYName[i+75708+10813].lower():
						features["findTrueEntity"][i] = 1
					if curRelation == allRelationY[i+75708+10813][3:]:
						features["findTrueRelation"][i] = 1
					findAns = 1
					features["findAns"][i] = 1
					if entityCandidates[i][j] == allEntityYName[i+75708+10813].lower() and curRelation == allRelationY[i+75708+10813][3:]:
						correctAns = 1
						features["correct"][i] = 1
					break
			if findAns == 1:
				break
		if findAns == 1:
			break
	if findAns == 0:
		features["useFulTop1Entity"][i] = 1
		features["useFulTop1Relation"][i] = 1
		if entityCandidates[i][0] == allEntityYName[i+75708+10813].lower():
			features["findTrueEntity"][i] = 1
		if relationIndexMap[argmaxPredictionTopN[0]] == allRelationY[i+75708+10813][3:]:
			features["findTrueRelation"][i] = 1
		features["numOfEntityLinkings"][i] = len(entityNameMap[entityCandidates[i][0]])
	correctCnt += correctAns
	output.write(allX[i+75708+10813] + "\t" + str(features["correct"][i]) + "\n")

print correctCnt*1.0/21632
output.close()

features.to_csv(path_or_buf='features.csv', sep='\t', index=False, header=True)

