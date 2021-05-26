import csv
import numpy as np
from pomegranate import *
import matplotlib.pyplot as plt

input1 = csv.reader(open('datos/2-Nivel.csv','r'))

pathTrain = []

nivel1Train = []
nivel2Train = []
nivel3Train = []
nivel4Train = []
nivel1Dic = {'': 0, None: 0}
nivel2Dic = {'': 0, None: 0}
nivel3Dic = {'': 0, None: 0}
nivel4Dic = {'': 0, None: 0}


typesTrain = []
typesDic = {'PAGE': 0, 'EVENT': 1}
lenTrain = []

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

	i+=1

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

for n in range (len(nivel1Train)):
	nivel = str(typesTrain[n]) + nivel1Train[n] + nivel2Train[n] + nivel3Train[n] + nivel4Train[n]
	pathTrain.append(int(nivel))

seq = np.array([pathTrain])

prob = [0.25, 0.5, 0.75]

for h in range (1,8):

	for l in prob:

		for k in prob:

			model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=h, X=seq, algorithm='baum-welch', edge_inertia=l, distribution_inertia=k)
			model.bake()
