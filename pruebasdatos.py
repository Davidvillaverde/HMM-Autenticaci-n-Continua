import csv
import numpy as np
from pomegranate import *
import matplotlib.pyplot as plt

input1 = csv.reader(open('datos/1-Nivel.csv','r'))
input2 = csv.reader(open('datos/2-Nivel.csv','r'))
input3 = csv.reader(open('datos/3-Nivel.csv','r'))
input4 = csv.reader(open('datos/4-Nivel.csv','r'))
input5 = csv.reader(open('datos/5-Nivel.csv','r'))

# pathTrain = []
# pathTest = []
# pathInt = []
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
nivel1Dic = {'': '0'}
nivel2Dic = {'': '0'}
nivel3Dic = {'': '0'}
nivel4Dic = {'': '0'}


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
					nivel1Dic[row[8]] = len(nivel1Dic)		
				nivel1Train.append(nivel1Dic[row[8]])
				if (row[9] not in nivel2Dic):
					nivel2Dic[row[9]] = len(nivel2Dic)		
				nivel2Train.append(nivel2Dic[row[9]])
				if (row[10] not in nivel3Dic):
					nivel3Dic[row[10]] = len(nivel3Dic)		
				nivel3Train.append(nivel3Dic[row[10]])
				if (row[11] not in nivel4Dic):
					nivel4Dic[row[11]] = len(nivel4Dic)		
				nivel4Train.append(nivel4Dic[row[11]])
			elif (row[6] == 'EVENT'):
				if (row[12] == None):
					nivel1Train.append(nivel1Dic[''])
				elif (row[12] not in nivel1Dic):
					nivel1Dic[row[12]] = len(nivel1Dic)		
					nivel1Train.append(nivel1Dic[row[12]])
				else:
					nivel1Train.append(nivel1Dic[row[12]])
				if (row[13] == None):
					nivel2Train.append(nivel2Dic[''])
				elif (row[13] not in nivel2Dic):
					nivel2Dic[row[13]] = len(nivel2Dic)		
					nivel2Train.append(nivel2Dic[row[13]])
				else:
					nivel2Train.append(nivel2Dic[row[13]])
				if (row[14] == None):
					nivel3Train.append(nivel3Dic[''])
				elif (row[14] not in nivel3Dic):
					nivel3Dic[row[14]] = len(nivel3Dic)		
					nivel3Train.append(nivel3Dic[row[14]])
				else:
					nivel3Train.append(nivel3Dic[row[14]])
				if (row[15] == None):
					nivel4Train.append(nivel4Dic[''])
				elif (row[15] not in nivel4Dic):
					nivel4Dic[row[15]] = len(nivel4Dic)		
					nivel4Train.append(nivel4Dic[row[15]])
				else:
					nivel4Train.append(nivel4Dic[row[15]])
			else:
				print('Nuevo tipo distinto')
				break

			if (tempId != row[0] ):
				tempId = row[0]
				lenTrain.append(int(row[3]))

			typesTrain.append(typesDic[row[6]])

		else:
			if (row[6] == 'PAGE'):
				if (row[8] not in nivel1Dic):
					nivel1Dic[row[8]] = len(nivel1Dic)		
				nivel1Test.append(nivel1Dic[row[8]])
				if (row[9] not in nivel2Dic):
					nivel2Dic[row[9]] = len(nivel2Dic)		
				nivel2Test.append(nivel2Dic[row[9]])
				if (row[10] not in nivel3Dic):
					nivel3Dic[row[10]] = len(nivel3Dic)		
				nivel3Test.append(nivel3Dic[row[10]])
				if (row[11] not in nivel4Dic):
					nivel4Dic[row[11]] = len(nivel4Dic)		
				nivel4Test.append(nivel4Dic[row[11]])
			elif (row[6] == 'EVENT'):
				if (row[12] == None):
					nivel1Test.append(nivel1Dic[''])
				elif (row[12] not in nivel1Dic):
					nivel1Dic[row[12]] = len(nivel1Dic)		
					nivel1Test.append(nivel1Dic[row[12]])
				else:
					nivel1Test.append(nivel1Dic[row[12]])
				if (row[13] == None):
					nivel2Test.append(nivel2Dic[''])
				elif (row[13] not in nivel2Dic):
					nivel2Dic[row[13]] = len(nivel2Dic)		
					nivel2Test.append(nivel2Dic[row[13]])
				else:
					nivel2Test.append(nivel2Dic[row[13]])
				if (row[14] == None):
					nivel3Test.append(nivel3Dic[''])
				elif (row[14] not in nivel3Dic):
					nivel3Dic[row[14]] = len(nivel3Dic)		
					nivel3Test.append(nivel3Dic[row[14]])
				else:
					nivel3Test.append(nivel3Dic[row[14]])
				if (row[15] == None):
					nivel4Test.append(nivel4Dic[''])
				elif (row[15] not in nivel4Dic):
					nivel4Dic[row[15]] = len(nivel4Dic)		
					nivel4Test.append(nivel4Dic[row[15]])
				else:
					nivel4Test.append(nivel4Dic[row[15]])
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
				if (row[8] not in nivel1Dic):
					nivel1Dic[row[8]] = len(nivel1Dic)		
				nivel1Int.append(nivel1Dic[row[8]])
				if (row[9] not in nivel2Dic):
					nivel2Dic[row[9]] = len(nivel2Dic)		
				nivel2Int.append(nivel2Dic[row[9]])
				if (row[10] not in nivel3Dic):
					nivel3Dic[row[10]] = len(nivel3Dic)		
				nivel3Int.append(nivel3Dic[row[10]])
				if (row[11] not in nivel4Dic):
					nivel4Dic[row[11]] = len(nivel4Dic)		
				nivel4Int.append(nivel4Dic[row[11]])
			elif (row[6] == 'EVENT'):
				if (row[12] == None):
					nivel1Int.append(nivel1Dic[''])
				elif (row[12] not in nivel1Dic):
					nivel1Dic[row[12]] = len(nivel1Dic)		
					nivel1Int.append(nivel1Dic[row[12]])
				else:
					nivel1Int.append(nivel1Dic[row[12]])
				if (row[13] == None):
					nivel2Int.append(nivel2Dic[''])
				elif (row[13] not in nivel2Dic):
					nivel2Dic[row[13]] = len(nivel2Dic)		
					nivel2Int.append(nivel2Dic[row[13]])
				else:
					nivel2Int.append(nivel2Dic[row[13]])
				if (row[14] == None):
					nivel3Int.append(nivel3Dic[''])
				elif (row[14] not in nivel3Dic):
					nivel3Dic[row[14]] = len(nivel3Dic)		
					nivel3Int.append(nivel3Dic[row[14]])
				else:
					nivel3Int.append(nivel3Dic[row[14]])
				if (row[15] == None):
					nivel4Int.append(nivel4Dic[''])
				elif (row[15] not in nivel4Dic):
					nivel4Dic[row[15]] = len(nivel4Dic)		
					nivel4Int.append(nivel4Dic[row[15]])
				else:
					nivel4Int.append(nivel4Dic[row[15]])
			else:
				print('Nuevo tipo distinto')
				break

			if (tempId != row[0] ):
				tempId = row[0]
				lenInt.append(int(row[3]))

			typesInt.append(typesDic[row[6]])

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
					nivel1Dic[row[8]] = len(nivel1Dic)		
				nivel1Int.append(nivel1Dic[row[8]])
				if (row[9] not in nivel2Dic):
					nivel2Dic[row[9]] = len(nivel2Dic)		
				nivel2Int.append(nivel2Dic[row[9]])
				if (row[10] not in nivel3Dic):
					nivel3Dic[row[10]] = len(nivel3Dic)		
				nivel3Int.append(nivel3Dic[row[10]])
				if (row[11] not in nivel4Dic):
					nivel4Dic[row[11]] = len(nivel4Dic)		
				nivel4Int.append(nivel4Dic[row[11]])
			elif (row[6] == 'EVENT'):
				if (row[12] == None):
					nivel1Int.append(nivel1Dic[''])
				elif (row[12] not in nivel1Dic):
					nivel1Dic[row[12]] = len(nivel1Dic)		
					nivel1Int.append(nivel1Dic[row[12]])
				else:
					nivel1Int.append(nivel1Dic[row[12]])
				if (row[13] == None):
					nivel2Int.append(nivel2Dic[''])
				elif (row[13] not in nivel2Dic):
					nivel2Dic[row[13]] = len(nivel2Dic)		
					nivel2Int.append(nivel2Dic[row[13]])
				else:
					nivel2Int.append(nivel2Dic[row[13]])
				if (row[14] == None):
					nivel3Int.append(nivel3Dic[''])
				elif (row[14] not in nivel3Dic):
					nivel3Dic[row[14]] = len(nivel3Dic)		
					nivel3Int.append(nivel3Dic[row[14]])
				else:
					nivel3Int.append(nivel3Dic[row[14]])
				if (row[15] == None):
					nivel4Int.append(nivel4Dic[''])
				elif (row[15] not in nivel4Dic):
					nivel4Dic[row[15]] = len(nivel4Dic)		
					nivel4Int.append(nivel4Dic[row[15]])
				else:
					nivel4Int.append(nivel4Dic[row[15]])
			else:
				print('Nuevo tipo distinto')
				break

			if (tempId != row[0] ):
				tempId = row[0]
				lenInt.append(int(row[3]))

			typesInt.append(typesDic[row[6]])

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
					nivel1Dic[row[8]] = len(nivel1Dic)		
				nivel1Int.append(nivel1Dic[row[8]])
				if (row[9] not in nivel2Dic):
					nivel2Dic[row[9]] = len(nivel2Dic)		
				nivel2Int.append(nivel2Dic[row[9]])
				if (row[10] not in nivel3Dic):
					nivel3Dic[row[10]] = len(nivel3Dic)		
				nivel3Int.append(nivel3Dic[row[10]])
				if (row[11] not in nivel4Dic):
					nivel4Dic[row[11]] = len(nivel4Dic)		
				nivel4Int.append(nivel4Dic[row[11]])
			elif (row[6] == 'EVENT'):
				if (row[12] == None):
					nivel1Int.append(nivel1Dic[''])
				elif (row[12] not in nivel1Dic):
					nivel1Dic[row[12]] = len(nivel1Dic)		
					nivel1Int.append(nivel1Dic[row[12]])
				else:
					nivel1Int.append(nivel1Dic[row[12]])
				if (row[13] == None):
					nivel2Int.append(nivel2Dic[''])
				elif (row[13] not in nivel2Dic):
					nivel2Dic[row[13]] = len(nivel2Dic)		
					nivel2Int.append(nivel2Dic[row[13]])
				else:
					nivel2Int.append(nivel2Dic[row[13]])
				if (row[14] == None):
					nivel3Int.append(nivel3Dic[''])
				elif (row[14] not in nivel3Dic):
					nivel3Dic[row[14]] = len(nivel3Dic)		
					nivel3Int.append(nivel3Dic[row[14]])
				else:
					nivel3Int.append(nivel3Dic[row[14]])
				if (row[15] == None):
					nivel4Int.append(nivel4Dic[''])
				elif (row[15] not in nivel4Dic):
					nivel4Dic[row[15]] = len(nivel4Dic)		
					nivel4Int.append(nivel4Dic[row[15]])
				else:
					nivel4Int.append(nivel4Dic[row[15]])
			else:
				print('Nuevo tipo distinto')
				break

			if (tempId != row[0] ):
				tempId = row[0]
				lenInt.append(int(row[3]))

			typesInt.append(typesDic[row[6]])

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
					nivel1Dic[row[8]] = len(nivel1Dic)		
				nivel1Int.append(nivel1Dic[row[8]])
				if (row[9] not in nivel2Dic):
					nivel2Dic[row[9]] = len(nivel2Dic)		
				nivel2Int.append(nivel2Dic[row[9]])
				if (row[10] not in nivel3Dic):
					nivel3Dic[row[10]] = len(nivel3Dic)		
				nivel3Int.append(nivel3Dic[row[10]])
				if (row[11] not in nivel4Dic):
					nivel4Dic[row[11]] = len(nivel4Dic)		
				nivel4Int.append(nivel4Dic[row[11]])
			elif (row[6] == 'EVENT'):
				if (row[12] == None):
					nivel1Int.append(nivel1Dic[''])
				elif (row[12] not in nivel1Dic):
					nivel1Dic[row[12]] = len(nivel1Dic)		
					nivel1Int.append(nivel1Dic[row[12]])
				else:
					nivel1Int.append(nivel1Dic[row[12]])
				if (row[13] == None):
					nivel2Int.append(nivel2Dic[''])
				elif (row[13] not in nivel2Dic):
					nivel2Dic[row[13]] = len(nivel2Dic)		
					nivel2Int.append(nivel2Dic[row[13]])
				else:
					nivel2Int.append(nivel2Dic[row[13]])
				if (row[14] == None):
					nivel3Int.append(nivel3Dic[''])
				elif (row[14] not in nivel3Dic):
					nivel3Dic[row[14]] = len(nivel3Dic)		
					nivel3Int.append(nivel3Dic[row[14]])
				else:
					nivel3Int.append(nivel3Dic[row[14]])
				if (row[15] == None):
					nivel4Int.append(nivel4Dic[''])
				elif (row[15] not in nivel4Dic):
					nivel4Dic[row[15]] = len(nivel4Dic)		
					nivel4Int.append(nivel4Dic[row[15]])
				else:
					nivel4Int.append(nivel4Dic[row[15]])
			else:
				print('Nuevo tipo distinto')
				break

			if (tempId != row[0] ):
				tempId = row[0]
				lenInt.append(int(row[3]))

			typesInt.append(typesDic[row[6]])

	i+=1

print('nivel 1')
print(nivel1Dic)
print('nivel 2')
print(nivel2Dic)
print('nivel 3')
print(nivel3Dic)
print('nivel 4')
print(nivel4Dic)



