import csv
import numpy as np
from pomegranate import *
import matplotlib.pyplot as plt

input1 = csv.reader(open('datos/1-path.csv','r'))
input2 = csv.reader(open('datos/2-path.csv','r'))
input3 = csv.reader(open('datos/3-path.csv','r'))
input4 = csv.reader(open('datos/4-path.csv','r'))
input5 = csv.reader(open('datos/5-path.csv','r'))

pathTrain = []
pathTest = []
pathInt = []
pathDic = {}
typesTrain = []
typesTest = []
typesInt = []
typesDic = {'PAGE': 0, 'EVENT': 1}
lenTrain = []
lenTest = []
lenInt = []

nvis1 = []
i=0
tempId = 0

for row in input1:
	if (i > 0):
		nvis1.append(row[1])

		if ((int(row[1])-int(nvis1[0])) < 200):
			if (row[6] == 'PAGE'):
				if (row[7] not in pathDic):
					pathDic[row[7]] = len(pathDic)		
				pathTrain.append(pathDic[row[7]])
			elif (row[6] == 'EVENT'):
				if (row[9] not in pathDic):
					pathDic[row[9]] = len(pathDic)		
				pathTrain.append(pathDic[row[9]])
			else:
				print('Nuevo tipo distinto')
				break

			if (tempId != row[0] ):
				tempId = row[0]
				lenTrain.append(int(row[3]))

			typesTrain.append(typesDic[row[6]])

		else:
			if (row[6] == 'PAGE'):
				if (row[7] not in pathDic):
					pathDic[row[7]] = len(pathDic)		
				pathTest.append(pathDic[row[7]])
			elif (row[6] == 'EVENT'):
				if (row[9] not in pathDic):
					pathDic[row[9]] = len(pathDic)		
				pathTest.append(pathDic[row[9]])
			else:
				print('Nuevo tipo distinto')
				break

			if (tempId != row[0] ):
				tempId = row[0]
				lenTest.append(int(row[3]))

			typesTest.append(typesDic[row[6]])

	i+=1

nvis1.clear()
i=0
tempId = 0

for row in input2:
	if (i > 0):
		nvis1.append(row[1])

		if ((int(row[1])-int(nvis1[0])) < 50):
			if (row[6] == 'PAGE'):
				if (row[7] not in pathDic):
					pathDic[row[7]] = len(pathDic)		
				pathInt.append(pathDic[row[7]])
			elif (row[6] == 'EVENT'):
				if (row[9] not in pathDic):
					pathDic[row[9]] = len(pathDic)		
				pathInt.append(pathDic[row[9]])
			else:
				print('Nuevo tipo distinto')
				break

			if (tempId != row[0] ):
				tempId = row[0]
				lenInt.append(int(row[3]))

			typesInt.append(typesDic[row[6]])

		else:
			break
	i+=1

nvis1.clear()
i=0
tempId = 0

for row in input3:
	if (i > 0):
		nvis1.append(row[1])

		if ((int(row[1])-int(nvis1[0])) < 50):
			if (row[6] == 'PAGE'):
				if (row[7] not in pathDic):
					pathDic[row[7]] = len(pathDic)		
				pathInt.append(pathDic[row[7]])
			elif (row[6] == 'EVENT'):
				if (row[9] not in pathDic):
					pathDic[row[9]] = len(pathDic)		
				pathInt.append(pathDic[row[9]])
			else:
				print('Nuevo tipo distinto')
				break

			if (tempId != row[0] ):
				tempId = row[0]
				lenInt.append(int(row[3]))

			typesInt.append(typesDic[row[6]])

		else:
			break

	i+=1

nvis1.clear()
i=0
tempId = 0

for row in input4:
	if (i > 0):
		nvis1.append(row[1])

		if ((int(row[1])-int(nvis1[0])) < 50):
			if (row[6] == 'PAGE'):
				if (row[7] not in pathDic):
					pathDic[row[7]] = len(pathDic)		
				pathInt.append(pathDic[row[7]])
			elif (row[6] == 'EVENT'):
				if (row[9] not in pathDic):
					pathDic[row[9]] = len(pathDic)		
				pathInt.append(pathDic[row[9]])
			else:
				print('Nuevo tipo distinto')
				break

			if (tempId != row[0] ):
				tempId = row[0]
				lenInt.append(int(row[3]))

			typesInt.append(typesDic[row[6]])

		else:
			break

	i+=1

nvis1.clear()
i=0
tempId = 0

for row in input5:
	if (i > 0):
		nvis1.append(row[1])

		if ((int(row[1])-int(nvis1[0])) < 50):
			if (row[6] == 'PAGE'):
				if (row[7] not in pathDic):
					pathDic[row[7]] = len(pathDic)		
				pathInt.append(pathDic[row[7]])
			elif (row[6] == 'EVENT'):
				if (row[9] not in pathDic):
					pathDic[row[9]] = len(pathDic)		
				pathInt.append(pathDic[row[9]])
			else:
				print('Nuevo tipo distinto')
				break

			if (tempId != row[0] ):
				tempId = row[0]
				lenInt.append(int(row[3]))

			typesInt.append(typesDic[row[6]])

		else:
			break

	i+=1

#print(lenTrain)
#print(lenTest)
seq = np.array((typesTrain,pathTrain))
#print(seq)

prob = [0.25, 0.5, 0.75]

for h in range (1,18):

	for l in prob:

		for k in prob:

			#model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=5, X=seq)
			model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=h, X=seq, algorithm='baum-welch', edge_inertia=l, distribution_inertia=k)
			model.bake()
			#print(model.viterbi(np.array((types[0:3926],path[0:3926]))))
			counterTrain = 0
			scores_Train = []

			for x in lenTrain:
				if (int(x) > 1):
					scoreTrain = model.log_probability(np.array((typesTrain[counterTrain:counterTrain+x-1],pathTrain[counterTrain:counterTrain+x-1])))
					scores_Train.append(scoreTrain)
				counterTrain += x

			counterTest = 0
			scores_Test = []

			for x in lenTest:
				if (int(x) > 1):
					scoreTest = model.log_probability(np.array((typesTest[counterTest:counterTest+x-1],pathTest[counterTest:counterTest+x-1])))
					scores_Test.append(scoreTest)
				counterTest += x

			counterInt = 0
			scores_Int = []

			for x in lenInt:
				if (int(x) > 1):
					scoreInt = model.log_probability(np.array((typesInt[counterInt:counterInt+x-1],pathInt[counterInt:counterInt+x-1])))
					scores_Int.append(scoreInt)
				counterInt += x

			length_train = len(scores_Train)
			length_val = len(scores_Test) + length_train
			length_int = len(scores_Int) + length_val

			plt.figure(figsize=(9,7))
			plt.scatter(np.arange(length_train), scores_Train, c='b', label='trainset')
			plt.scatter(np.arange(length_train, length_val), scores_Test, c='r', label='testset - original')
			plt.scatter(np.arange(length_val, length_int), scores_Int, c='g', label='testset - intruso')
			plt.title(label="N comp: "+str(h)+" Edge:"+str(l)+" Distribution:"+str(k))
			plt.savefig("img/pomeConInercia_comp"+str(h)+"_edge"+str(l)+"Dist"+str(k)+".png")
			plt.close()
#plt.show()
#print(model.log_probability(np.array((typesTest,pathTest))))


