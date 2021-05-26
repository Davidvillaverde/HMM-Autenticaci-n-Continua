import csv
import numpy as np
from pomegranate import *
import matplotlib.pyplot as plt

input1 = csv.reader(open('datos/1-Nivel.csv','r'))
input2 = csv.reader(open('datos/2-Nivel.csv','r'))
input3 = csv.reader(open('datos/3-Nivel.csv','r'))
input4 = csv.reader(open('datos/4-Nivel.csv','r'))
input5 = csv.reader(open('datos/5-Nivel.csv','r'))

pathTrain = []
pathTest = []
pathInt = []
# pathDic = {}
nivel1Train = []
nivel2Train = []
nivel3Train = []
nivel4Train = []
nivel1Test= []
nivel2Test = []
nivel3Test = []
nivel4Test = []
nivel1Int = []
nivel2Int = []
nivel3Int = []
nivel4Int = []
nivel1Dic = {'': 0, None: 0}
nivel2Dic = {'': 0, None: 0}
nivel3Dic = {'': 0, None: 0}
nivel4Dic = {'': 0, None: 0}
timeTrain = []
timeTest = []
timeInt = []

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
				if (row[8] not in nivel1Dic):
					nivel1Dic[row[8]] = len(nivel1Dic)-1		
				nivel1Train.append(nivel1Dic[row[8]])

				if (row[9] not in nivel2Dic):
					nivel2Dic[row[9]] = len(nivel2Dic)-1	
				nivel2Train.append(nivel2Dic[row[9]])

				if (row[10] not in nivel3Dic):
					nivel3Dic[row[10]] = len(nivel3Dic)-1		
				nivel3Train.append(nivel3Dic[row[10]])

				if (row[11] not in nivel4Dic):
					nivel4Dic[row[11]] = len(nivel4Dic)-1		
				nivel4Train.append(nivel4Dic[row[11]])

			elif (row[6] == 'EVENT'):
				if (row[12] not in nivel1Dic):
					nivel1Dic[row[12]] = len(nivel1Dic)-1		
				nivel1Train.append(nivel1Dic[row[12]])

				if (row[13] not in nivel2Dic):
					nivel2Dic[row[13]] = len(nivel2Dic)-1		
				nivel2Train.append(nivel2Dic[row[13]])

				if (row[14] not in nivel3Dic):
					nivel3Dic[row[14]] = len(nivel3Dic)-1		
				nivel3Train.append(nivel3Dic[row[14]])

				if (row[15] not in nivel4Dic):
					nivel4Dic[row[15]] = len(nivel4Dic)-1		
				nivel4Train.append(nivel4Dic[row[15]])

			else:
				print('Nuevo tipo distinto')
				break

			if (tempId != row[0] ):
				tempId = row[0]
				lenTrain.append(int(row[3]))

			typesTrain.append(typesDic[row[6]])
			timeTrain.append(int(row[2]))

		else:
			if (row[6] == 'PAGE'):

				if (row[8] not in nivel1Dic):
					nivel1Dic[row[8]] = len(nivel1Dic)-1		
				nivel1Test.append(nivel1Dic[row[8]])

				if (row[9] not in nivel2Dic):
					nivel2Dic[row[9]] = len(nivel2Dic)-1		
				nivel2Test.append(nivel2Dic[row[9]])

				if (row[10] not in nivel3Dic):
					nivel3Dic[row[10]] = len(nivel3Dic)-1		
				nivel3Test.append(nivel3Dic[row[10]])

				if (row[11] not in nivel4Dic):
					nivel4Dic[row[11]] = len(nivel4Dic)-1		
				nivel4Test.append(nivel4Dic[row[11]])

			elif (row[6] == 'EVENT'):

				if (row[12] not in nivel1Dic):
					nivel1Dic[row[12]] = len(nivel1Dic)-1		
				nivel1Test.append(nivel1Dic[row[12]])

				if (row[13] not in nivel2Dic):
					nivel2Dic[row[13]] = len(nivel2Dic)-1		
				nivel2Test.append(nivel2Dic[row[13]])

				if (row[14] not in nivel3Dic):
					nivel3Dic[row[14]] = len(nivel3Dic)-1		
				nivel3Test.append(nivel3Dic[row[14]])

				if (row[15] not in nivel4Dic):
					nivel4Dic[row[15]] = len(nivel4Dic)-1		
				nivel4Test.append(nivel4Dic[row[15]])

			else:
				print('Nuevo tipo distinto')
				break

			if (tempId != row[0] ):
				tempId = row[0]
				lenTest.append(int(row[3]))

			typesTest.append(typesDic[row[6]])
			timeTest.append(int(row[2]))

	i+=1

nvis1.clear()
i=0
tempId = 0

for row in input2:
	if (i > 0):
		nvis1.append(row[1])

		if ((int(row[1])-int(nvis1[0])) < 50):
			if (row[6] == 'PAGE'):

				if (row[8] not in nivel1Dic):
					nivel1Dic[row[8]] = len(nivel1Dic)-1		
				nivel1Int.append(nivel1Dic[row[8]])

				if (row[9] not in nivel2Dic):
					nivel2Dic[row[9]] = len(nivel2Dic)-1		
				nivel2Int.append(nivel2Dic[row[9]])

				if (row[10] not in nivel3Dic):
					nivel3Dic[row[10]] = len(nivel3Dic)-1		
				nivel3Int.append(nivel3Dic[row[10]])

				if (row[11] not in nivel4Dic):
					nivel4Dic[row[11]] = len(nivel4Dic)-1		
				nivel4Int.append(nivel4Dic[row[11]])

			elif (row[6] == 'EVENT'):

				if (row[12] not in nivel1Dic):
					nivel1Dic[row[12]] = len(nivel1Dic)-1		
				nivel1Int.append(nivel1Dic[row[12]])

				if (row[13] not in nivel2Dic):
					nivel2Dic[row[13]] = len(nivel2Dic)-1		
				nivel2Int.append(nivel2Dic[row[13]])

				if (row[14] not in nivel3Dic):
					nivel3Dic[row[14]] = len(nivel3Dic)-1		
				nivel3Int.append(nivel3Dic[row[14]])

				if (row[15] not in nivel4Dic):
					nivel4Dic[row[15]] = len(nivel4Dic)-1		
				nivel4Int.append(nivel4Dic[row[15]])

			else:
				print('Nuevo tipo distinto')
				break

			if (tempId != row[0] ):
				tempId = row[0]
				lenInt.append(int(row[3]))

			typesInt.append(typesDic[row[6]])
			timeInt.append(int(row[2]))

	i+=1

nvis1.clear()
i=0
tempId = 0

for row in input3:
	if (i > 0):
		nvis1.append(row[1])

		if ((int(row[1])-int(nvis1[0])) < 50):
			if (row[6] == 'PAGE'):

				if (row[8] not in nivel1Dic):
					nivel1Dic[row[8]] = len(nivel1Dic)-1		
				nivel1Int.append(nivel1Dic[row[8]])

				if (row[9] not in nivel2Dic):
					nivel2Dic[row[9]] = len(nivel2Dic)-1		
				nivel2Int.append(nivel2Dic[row[9]])

				if (row[10] not in nivel3Dic):
					nivel3Dic[row[10]] = len(nivel3Dic)-1		
				nivel3Int.append(nivel3Dic[row[10]])

				if (row[11] not in nivel4Dic):
					nivel4Dic[row[11]] = len(nivel4Dic)-1		
				nivel4Int.append(nivel4Dic[row[11]])

			elif (row[6] == 'EVENT'):

				if (row[12] not in nivel1Dic):
					nivel1Dic[row[12]] = len(nivel1Dic)-1		
				nivel1Int.append(nivel1Dic[row[12]])

				if (row[13] not in nivel2Dic):
					nivel2Dic[row[13]] = len(nivel2Dic)-1		
				nivel2Int.append(nivel2Dic[row[13]])

				if (row[14] not in nivel3Dic):
					nivel3Dic[row[14]] = len(nivel3Dic)-1		
				nivel3Int.append(nivel3Dic[row[14]])

				if (row[15] not in nivel4Dic):
					nivel4Dic[row[15]] = len(nivel4Dic)-1		
				nivel4Int.append(nivel4Dic[row[15]])

			else:
				print('Nuevo tipo distinto')
				break

			if (tempId != row[0] ):
				tempId = row[0]
				lenInt.append(int(row[3]))

			typesInt.append(typesDic[row[6]])
			timeInt.append(int(row[2]))

	i+=1

nvis1.clear()
i=0
tempId = 0

for row in input4:
	if (i > 0):
		nvis1.append(row[1])

		if ((int(row[1])-int(nvis1[0])) < 50):
			if (row[6] == 'PAGE'):

				if (row[8] not in nivel1Dic):
					nivel1Dic[row[8]] = len(nivel1Dic)-1		
				nivel1Int.append(nivel1Dic[row[8]])

				if (row[9] not in nivel2Dic):
					nivel2Dic[row[9]] = len(nivel2Dic)-1		
				nivel2Int.append(nivel2Dic[row[9]])

				if (row[10] not in nivel3Dic):
					nivel3Dic[row[10]] = len(nivel3Dic)-1		
				nivel3Int.append(nivel3Dic[row[10]])

				if (row[11] not in nivel4Dic):
					nivel4Dic[row[11]] = len(nivel4Dic)-1		
				nivel4Int.append(nivel4Dic[row[11]])

			elif (row[6] == 'EVENT'):

				if (row[12] not in nivel1Dic):
					nivel1Dic[row[12]] = len(nivel1Dic)-1		
				nivel1Int.append(nivel1Dic[row[12]])

				if (row[13] not in nivel2Dic):
					nivel2Dic[row[13]] = len(nivel2Dic)-1		
				nivel2Int.append(nivel2Dic[row[13]])

				if (row[14] not in nivel3Dic):
					nivel3Dic[row[14]] = len(nivel3Dic)-1		
				nivel3Int.append(nivel3Dic[row[14]])

				if (row[15] not in nivel4Dic):
					nivel4Dic[row[15]] = len(nivel4Dic)-1		
				nivel4Int.append(nivel4Dic[row[15]])

			else:
				print('Nuevo tipo distinto')
				break

			if (tempId != row[0] ):
				tempId = row[0]
				lenInt.append(int(row[3]))

			typesInt.append(typesDic[row[6]])
			timeInt.append(int(row[2]))

	i+=1

nvis1.clear()
i=0
tempId = 0

for row in input5:
	if (i > 0):
		nvis1.append(row[1])

		if ((int(row[1])-int(nvis1[0])) < 50):
			if (row[6] == 'PAGE'):

				if (row[8] not in nivel1Dic):
					nivel1Dic[row[8]] = len(nivel1Dic)-1		
				nivel1Int.append(nivel1Dic[row[8]])

				if (row[9] not in nivel2Dic):
					nivel2Dic[row[9]] = len(nivel2Dic)-1		
				nivel2Int.append(nivel2Dic[row[9]])

				if (row[10] not in nivel3Dic):
					nivel3Dic[row[10]] = len(nivel3Dic)-1		
				nivel3Int.append(nivel3Dic[row[10]])

				if (row[11] not in nivel4Dic):
					nivel4Dic[row[11]] = len(nivel4Dic)-1		
				nivel4Int.append(nivel4Dic[row[11]])

			elif (row[6] == 'EVENT'):

				if (row[12] not in nivel1Dic):
					nivel1Dic[row[12]] = len(nivel1Dic)-1		
				nivel1Int.append(nivel1Dic[row[12]])

				if (row[13] not in nivel2Dic):
					nivel2Dic[row[13]] = len(nivel2Dic)-1		
				nivel2Int.append(nivel2Dic[row[13]])

				if (row[14] not in nivel3Dic):
					nivel3Dic[row[14]] = len(nivel3Dic)-1		
				nivel3Int.append(nivel3Dic[row[14]])

				if (row[15] not in nivel4Dic):
					nivel4Dic[row[15]] = len(nivel4Dic)-1		
				nivel4Int.append(nivel4Dic[row[15]])

			else:
				print('Nuevo tipo distinto')
				break

			if (tempId != row[0] ):
				tempId = row[0]
				lenInt.append(int(row[3]))

			typesInt.append(typesDic[row[6]])
			timeInt.append(int(row[2]))

	i+=1

# print('nivel 1')
# print(nivel1Dic)
# print('nivel 2')
# print(nivel2Dic)
# print('nivel 3')
# print(nivel3Dic)
# print('nivel 4')
# print(nivel4Dic)

# print('nivel1')
# print(nivel1Train)
# print('nivel2')
# print(nivel2Train)
# print('nivel3')
# print(nivel3Train)
# print('nivel4')
# print(nivel4Train)

for n, q in enumerate(nivel1Train):
	if len(str(q)) == 1:
		nivel1Train[n] = '0'+ str(nivel1Train[n])
	else:
		nivel1Train[n] = str(nivel1Train[n])

for n, q in enumerate(nivel2Train):
	if len(str(q)) == 1:
		nivel2Train[n] = '0'+ str(nivel2Train[n])
	else:
		nivel2Train[n] = str(nivel2Train[n])

for n, q in enumerate(nivel3Train):
	if len(str(q)) == 1:
		nivel3Train[n] = '00'+ str(nivel3Train[n])
	elif len(str(q)) == 2:
		nivel3Train[n] = '0'+ str(nivel3Train[n])
	else:
		nivel3Train[n] = str(nivel3Train[n])

for n, q in enumerate(nivel4Train):
	if len(str(q)) == 1:
		nivel4Train[n] = '00'+ str(nivel4Train[n])
	elif len(str(q)) == 2:
		nivel4Train[n] = '0'+ str(nivel4Train[n])
	else:
		nivel4Train[n] = str(nivel4Train[n])

for n, q in enumerate(nivel1Test):
	if len(str(q)) == 1:
		nivel1Test[n] = '0'+ str(nivel1Test[n])
	else:
		nivel1Test[n] = str(nivel1Test[n])

for n, q in enumerate(nivel2Test):
	if len(str(q)) == 1:
		nivel2Test[n] = '0'+ str(nivel2Test[n])
	else:
		nivel2Test[n] = str(nivel2Test[n])

for n, q in enumerate(nivel3Test):
	if len(str(q)) == 1:
		nivel3Test[n] = '00'+ str(nivel3Test[n])
	elif len(str(q)) == 2:
		nivel3Test[n] = '0'+ str(nivel3Test[n])
	else:
		nivel3Test[n] = str(nivel3Test[n])

for n, q in enumerate(nivel4Test):
	if len(str(q)) == 1:
		nivel4Test[n] = '00'+ str(nivel4Test[n])
	elif len(str(q)) == 2:
		nivel4Test[n] = '0'+ str(nivel4Test[n])
	else:
		nivel4Test[n] = str(nivel4Test[n])

for n, q in enumerate(nivel1Int):
	if len(str(q)) == 1:
		nivel1Int[n] = '0'+ str(nivel1Int[n])
	else:
		nivel1Int[n] = str(nivel1Int[n])

for n, q in enumerate(nivel2Int):
	if len(str(q)) == 1:
		nivel2Int[n] = '0'+ str(nivel2Int[n])
	else:
		nivel2Int[n] = str(nivel2Int[n])

for n, q in enumerate(nivel3Int):
	if len(str(q)) == 1:
		nivel3Int[n] = '00'+ str(nivel3Int[n])
	elif len(str(q)) == 2:
		nivel3Int[n] = '0'+ str(nivel3Int[n])
	else:
		nivel3Int[n] = str(nivel3Int[n])

for n, q in enumerate(nivel4Int):
	if len(str(q)) == 1:
		nivel4Int[n] = '00'+ str(nivel4Int[n])
	elif len(str(q)) == 2:
		nivel4Int[n] = '0'+ str(nivel4Int[n])
	else:
		nivel4Int[n] = str(nivel4Int[n])

# print('nivel1')
# print(nivel1Train)
# print('nivel2')
# print(nivel2Train)
# print('nivel3')
# print(nivel3Train)
# print('nivel4')
# print(nivel4Train)

for n in range (len(nivel1Train)):
	nivel = str(typesTrain[n]) + nivel1Train[n] + nivel2Train[n] + nivel3Train[n] + nivel4Train[n]
	pathTrain.append(int(nivel))

for n in range (len(nivel1Test)):
	nivel = str(typesTest[n]) + nivel1Test[n] + nivel2Test[n] + nivel3Test[n] + nivel4Test[n]
	pathTest.append(int(nivel))

for n in range (len(nivel1Int)):
	nivel = str(typesInt[n]) + nivel1Int[n] + nivel2Int[n] + nivel3Int[n] + nivel4Int[n]
	pathInt.append(int(nivel))

# print('Train:')
# print(pathTrain)
# print('Test:')
# print(pathTest)
# print('Intruso:')
# print(pathInt)

seq = np.array((timeTrain,pathTrain))

bestPrecision = ""
bestRecall = ""
bestF1 = ""
bestAccuracy = ""
valorPrecision = 0
valorRecall = 0
valorF1 = 0
valorAccuracy = 0

prob = [0.25, 0.5, 0.75]

for h in range (1,18):

	for l in prob:

		for k in prob:

			model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=h, X=seq, algorithm='baum-welch', edge_inertia=l, distribution_inertia=k)
			model.bake()

			counterTrain = 0
			scores_Train = []

			for x in lenTrain:
				scoreTrain = model.log_probability(np.array((typesTrain[counterTrain:counterTrain+x-1],pathTrain[counterTrain:counterTrain+x-1])))
				scores_Train.append(scoreTrain)
				counterTrain += x

			counterTest = 0
			scores_Test = []

			for x in lenTest:
				scoreTest = model.log_probability(np.array((typesTest[counterTest:counterTest+x-1],pathTest[counterTest:counterTest+x-1])))
				scores_Test.append(scoreTest)
				counterTest += x

			counterInt = 0
			scores_Int = []

			for x in lenInt:
				scoreInt = model.log_probability(np.array((typesInt[counterInt:counterInt+x-1],pathInt[counterInt:counterInt+x-1])))
				scores_Int.append(scoreInt)
				counterInt += x

			stat = np.array([scores_Train], dtype="float")
			stat = np.append(stat,scores_Test)
			media = np.mean(stat)
			deviation = np.std(stat)
			umbSup = media + deviation
			umbInf = media - deviation
			cuentaSI = np.count_nonzero(np.logical_and(stat < umbSup, stat > umbInf))
			statNO = np.array([scores_Int], dtype="float")
			cuentaNO = np.count_nonzero(np.logical_and(statNO < umbSup, statNO > umbInf))

			TP = cuentaSI
			FN = stat.size - cuentaSI
			FP = cuentaNO
			TN = statNO.size - cuentaNO

			if ((TP+FP)==0 or (TP+FP)==0):
				precision = 0
				recall = 0
				F1 = 0
				accuracy = 0
			else:
				precision = TP / (TP + FP)
				recall = TP / (TP + FN)
				F1 = 2 * (precision * recall) / (precision + recall)
				accuracy = (TP + TN) / (TP + TN + FP + FN)

				if (precision > valorPrecision):
					valorPrecision = precision
					bestPrecision = "Comp: "+str(h)+" Edge: "+str(l)+" Dist: "+str(k)
				if (recall > valorRecall):
					valorRecall = recall
					bestRecall = "Comp: "+str(h)+" Edge: "+str(l)+" Dist: "+str(k)
				if (F1 > valorF1):
					valorF1 = F1
					bestF1 = "Comp: "+str(h)+" Edge: "+str(l)+" Dist: "+str(k)
				if (accuracy > valorAccuracy):
					valorAccuracy = accuracy
					bestAccuracy = "Comp: "+str(h)+" Edge: "+str(l)+" Dist: "+str(k)


			length_train = len(scores_Train)
			length_val = len(scores_Test) + length_train
			length_int = len(scores_Int) + length_val

			plt.figure(figsize=(9,7))
			plt.scatter(np.arange(length_train), scores_Train, c='b', label='Entrenamiento')
			plt.scatter(np.arange(length_train, length_val), scores_Test, c='r', label='Test original')
			plt.scatter(np.arange(length_val, length_int), scores_Int, c='g', label='Intruso')
			plt.title(label=" Media:"+str(media)[:6]+" Deviation:"+str(deviation)[:4]+" True Positive:"+str(cuentaSI)+"/"+str(stat.size)+" False Positive:"+str(cuentaNO)+"/"+str(statNO.size))
			plt.suptitle('Precision: '+str(precision)[:4]+' Recall: '+str(recall)[:4]+' F1: '+str(F1)[:4]+' Accuracy: '+str(accuracy)[:4])
			plt.xlabel('Sesión')
			plt.ylabel('Puntuación')
			plt.legend()
			plt.savefig("img/pome_Comp"+str(h)+"_Edge"+str(l)+"_Dist"+str(k)+".png")
			plt.close()

f = open("img/Best.txt", "w")
f.write("bestPrecision: " + bestPrecision)
f.write(" Valor: " + str(valorPrecision))
f.write("\r\nbestRecall: " + bestRecall)
f.write(" Valor: " + str(valorRecall))
f.write("\r\nbestF1: " + bestF1)
f.write(" Valor: " + str(valorF1))
f.write("\r\nbestAccuracy: " + bestAccuracy)
f.write(" Valor: " + str(valorAccuracy))
f.close()