# linreg.py
# Machine Learning - Linear Regression
# Hyoyup Chung
# 	data parameter selected for home price:
#		beds, baths, sqft


import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

Epsilon = 0.0000001

# main:
if __name__ == "__main__":

	# Preprocess Data and Clean Up
	csvData = pd.read_csv("data.csv", delimiter = ',')
	csvData = csvData.dropna(subset=['BEDS'])
	csvData = csvData.dropna(subset=['BATHS'])
	csvData = csvData.dropna(subset=['SQUARE FEET'])
	csvData = csvData.dropna(subset=['PRICE'])

	# Histogram for Square feet & Price
	f1 = plt.figure(1)
	plt.title('mean=red, median=green, std = black')
	plt.xlabel('sqft.')
	csvData['SQUARE FEET'].hist(bins=100)
	plt.axvline(csvData['SQUARE FEET'].mean(), color='r', linestyle='dashed', linewidth=3)
	plt.axvline(csvData['SQUARE FEET'].median(), color='g', linestyle='dashed', linewidth=3)
	plt.axvline(csvData['SQUARE FEET'].std(), color='k', linestyle='dashed', linewidth=3)

	f2 = plt.figure(2)
	plt.title('mean=red, median=green, std = black')
	plt.xlabel('Price (dollars)')
	csvData['PRICE'].hist(bins=100)
	plt.axvline(csvData['PRICE'].mean(), color='r', linestyle='dashed', linewidth=3)
	plt.axvline(csvData['PRICE'].median(), color='g', linestyle='dashed', linewidth=3)
	plt.axvline(csvData['PRICE'].std(), color='k', linestyle='dashed', linewidth=3)

	#plt.show()


	# Generate Data Set
	csvDataX = np.empty((0,4),float)
	csvDataY = np.empty((0,1),float)
	csvDataNum = 0
	for i, row in csvData.iterrows():
		newData = [[1.0, float(csvData['BEDS'][i]), float(csvData['BATHS'][i]), float(csvData['SQUARE FEET'][i])]]
		csvDataX = np.append(csvDataX, newData, axis=0)
		csvDataY = np.append(csvDataY, [float(csvData['PRICE'][i])])
		csvDataNum += 1

	# csv data distribution (344 left after clean up)
	# Training 	206/344
	# Validation 69/344
	# Testing 69/344

	# Training Data [0,206)
	TrainingX = np.zeros((206,4))
	TrainingY = np.zeros((206,1))
	for i in range(0,206):
		TrainingX[i] = csvDataX[i]#[:]
		TrainingY[i] = csvDataY[i]

	xT = np.transpose(TrainingX)
	xTx = np.matmul(xT,TrainingX)
	xTx_inv = np.linalg.inv(xTx)

	xTy = np.matmul(xT,TrainingY)

	theta = np.matmul(xTx_inv,xTy)

	print("Linear Model:")
	print("h_theta(x) = %f + %f x1 + %f x2 + %f x3" % (theta[0], theta[1],theta[2],theta[3]))
	print("\t//where x1 = # of beds, x2 = # of baths, x3 = sqft")

	# Validating Data [206, 275)
	ValidationX = np.zeros((69,4))
	ValidationY = np.zeros((69,1))
	for i in range(0,69):
		ValidationX[i] = csvDataX[i+206]
		ValidationY[i] = csvDataY[i+206]


	f3 = plt.figure(3)
	plt.title('Validation: price vs sqft')
	plt.xlabel('Sqft')
	plt.ylabel('Price')
	plt.scatter(csvData['SQUARE FEET'][206:275],csvData['PRICE'][206:275])
	
	bedavg = csvData['BEDS'][206:275].mean()
	bathavg = csvData['BATHS'][206:275].mean()

	# theta = 33309 + 53459*bedavg - 27173*bathavg + 228*x3
	#		= 159315.6 + 228*x3			, x3=sqft

	x_val = csvData['SQUARE FEET'][206:275]
	plt.plot(x_val,159315.6+228*x_val,'r-')

	# Testing Data [275,344]
	TestingX = np.zeros((69,4))
	TestingY = np.zeros((69,1))
	for i in range(0,69):
		TestingX[i] = csvDataX[i+275]
		TestingY[i] = csvDataY[i+275]

	# calculate RMSE
	sqSum = 0.0
	for i in range(0,69):
		Yprod = theta[0] + theta[1]*TestingX[i][1] + theta[2]*TestingX[i][2] + theta[3]*TestingX[i][3]
		partE = Yprod - TestingY[i]
		sqSum += (partE*partE)
	sqSum = math.sqrt(sqSum) 

	RMSE = sqSum/69.0
	print(RMSE) # prints ~ 17513
	plt.show()