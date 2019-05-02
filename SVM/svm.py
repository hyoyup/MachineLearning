# svm.py

# Machine Learning - SVM
# Hyoyup Chung
# 	- iris flower type data: iris.csv
#	- classification, setosa vs. versicolor
#	- Y_k defined as :
#			{ 1: setosa, -1: versicolor }

# Run in command-line:
#	python svm.py
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

learningRate = 0.01
C_regularization = 10.0
MaxIter = 1000

def classifier(w,x,b):
	wT = np.transpose(w)
	res = np.matmul(wT,x)+b
	print(res)
	if res > 0:
		return 1.0
	return -1.0

# dL/dLambda
def deriv_Lagrangean_Lambda(index, setData, curLambda):
	summation = 0.0
	for k in range(0,len(curLambda)):
		xk_T = np.transpose(np.transpose([setData[k][:4]])).astype(float)
		xi = np.transpose([setData[index][:4]]).astype(float)
		summation += curLambda[k]*curLambda[index]*np.matmul(xk_T,xi)[0][0]
	return 1.0-summation

def exclusiveRandom(low,high,exclude):
	nextRand = np.random.random_integers(low,high)
	while(nextRand!=exclude):
		nextRand = np.random.random_integers(low,high)
	return nextRand

def get_Y(index, setData):
	irisTypename = setData[index][4]
	if (irisTypename=="Iris-setosa"):
		return 1.0
	return -1.0 # "Iris-versicolor"

# returns support vector's index
def findSupportVector(yk, w, setData):
	vectors = []
	for k in range(0,len(setData)):
		if get_Y(k,setData)==yk:
			wT = np.transpose(w)
			x_k = np.transpose([setData[k][:4]]).astype(float)
			vectors.append(yk*np.matmul(wT, x_k)[0][0])
		else:
			vectors.append(float("inf"))
	return np.argmin(vectors)

def CompareResult(Result, TestSetData):
	numCorrect = 0
	for i in range(0,len(TestSetData)):
		if Result[i] == get_Y(i,TestSetData):
			numCorrect+=1
	return numCorrect / len(TestSetData)

# main:
if __name__ == "__main__":

	# Preprocess Data
	csvData = pd.read_csv("iris.csv", header=None, delimiter = ',')
	csvData.rename(columns={0:'sepal_len',1:'sepal_wid',2:'petal_len',3:'petal_wid',4:'iris_type'},inplace=True)

	# Generate Data Set
	setosaData = csvData[:50]
	versicolorData = csvData[50:100]
	train_setosa, test_setosa = train_test_split(setosaData, shuffle=False, test_size=0.30)
	train_versi, test_versi = train_test_split(versicolorData, shuffle=False, test_size=0.30)

	TrainData = np.concatenate((train_setosa, train_versi),axis=0)
	NumTrain = len(TrainData)
	TestData = np.concatenate((test_setosa, test_versi),axis=0)
	NumTest = len(TestData)

	Lambdas = np.random.uniform(0.0,C_regularization,size=NumTrain)

	#############################
	######## TRAINING ###########
	#############################
	for iter in range(0,MaxIter):
		i = np.random.random_integers(0,NumTrain-1)
		der_L = deriv_Lagrangean_Lambda(i,TrainData,Lambdas)

		# coordinate ascent to maximize L
		Lambdas[i] = Lambdas[i] + learningRate*der_L

		# clamping [0, C_regularization]
		if (Lambdas[i]<0.0 or Lambdas[i]>C_regularization):
			Lambdas[i] = max(min(Lambdas[i],C_regularization),0.0)

		# choose random j != i
		j = exclusiveRandom(0,NumTrain-1,i)

		sum_lambdaY = 0.0
		for k in range(0,NumTrain):
			if (k==j): 
				continue
			sum_lambdaY += Lambdas[k]*get_Y(k,TrainData)

		# calculate lambda_j
		Lambdas[j] = -1.0*get_Y(j,TrainData)*sum_lambdaY

		# clamp lambda_j [0, C_regularization]
		if (Lambdas[j]<0.0 or Lambdas[j]>C_regularization):
			Lambdas[j] = max(min(Lambdas[j],C_regularization),0.0)

	# print(Lambdas)

	# calculate w
	w = np.zeros((4,1),float)
	for i in range(0,NumTrain):
		temp = Lambdas[i]*get_Y(i,TrainData)*TrainData[i][:4]
		w += [[temp[0]],[temp[1]],[temp[2]],[temp[3]]]
	# print(w)

	# find the two support vectors	
	X_setosa_index = findSupportVector(1.0,w,TrainData)
	X_versicolor_index = findSupportVector(-1.0,w,TrainData)
	
	X_setosa = TrainData[X_setosa_index][:4].astype(float)
	X_versicolor = TrainData[X_versicolor_index][:4].astype(float)

	# calculate b
	wT = np.transpose(w)
	b = -0.5*(np.matmul(wT,X_setosa)+np.matmul(wT,X_versicolor))
	# print(b)

	#############################
	######### Testing ###########
	#############################
	TestResult = []
	for i in range(0,NumTest):
		xi = TestData[i][:4].astype(float)
		TestResult.append(classifier(w,xi,b))
	print("Test Result: ")
	print(TestResult)

	#############################
	######## Validating #########
	#############################
	accuracy = CompareResult(TestResult, TestData)
	print("Accuracy of SVM on the TestingData: %f\n" % (accuracy))
	print("W :")
	print(w)
	print("\nb : %f" % (b))

	# visualization
	f1 = plt.figure(1)
	plt.title('Iris Flower Types: Setosa(red) & Versicolor(blue)')
	plt.xlabel('Sepal_Length')
	plt.ylabel('Sepal_Width')

	Sum_petlen = 0.0
	Sum_petwid = 0.0
	for i in range(0,NumTest):
		xi = TestData[i][:4].astype(float)
		Sum_petlen += xi[2]
		Sum_petwid += xi[3]
		if TestResult[i]==1.0:
			plt.plot(xi[0],xi[1],'r+')
		else:
			plt.plot(xi[0],xi[1],'b+')
	pet_len_avg = Sum_petlen/NumTest
	pet_wid_avg = Sum_petwid/NumTest
	x = np.linspace(4,6)

	# not sure if this line is correct but idea was (Refer to README for details)
	# 	using w^T*x + b = 1, solve equation f(sepal_len) = sepal_wid
	plt.plot(x,(-b-w[3]*pet_wid_avg-w[2]*pet_len_avg-x*w[0])/w[1]-1 ,color='g')
	plt.show()