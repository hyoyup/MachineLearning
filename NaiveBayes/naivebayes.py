# naivebayes.py

# Machine Learning - Naive Bayes
# Hyoyup Chung
# 	spam/ham data - test-features/labels.txt, train-features/labels.txt 

# Run in command-line:
# 	python naivebayes.py [# of data = none, 50, 100, 400]   <- SEE README
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math, sys

train_data_feature = "train-features"
train_data_label = "train-labels"

test_data_feature = "test-features.txt"
test_data_label = "test-labels.txt"
# default set info when no training size specified
dataNum = 700
hamNum = 350
spamNum = 350
testDataNum = 260

def TestBayes(spamDict, hamDict, testSet):
	res = []
	curmsgNum = 1
	curprob_spam = 0
	curprob_ham = 0

	for row in testSet.iterrows():
		msgNum = int(row[1][0])
		wordIndex = int(row[1][1])
		# when one msg eval is complete
		if (curmsgNum != msgNum):
			curmsgNum += 1
			# last term ln(P(spam)), ln(P(ham)) 
			#	trivial since sets are 1:1
			curprob_ham += np.log(hamNum/dataNum)
			curprob_spam += np.log(spamNum/dataNum)
			if curprob_ham < curprob_spam:
				res.append(1)
			else:	
				res.append(0)
			curprob_ham = 0
			curprob_spam = 0
		P_xi_spam = (spamDict[wordIndex]+1) / (dataNum/2 + 2)
		P_xi_ham = (hamDict[wordIndex]+1) / (dataNum/2 + 2) 
		curprob_spam += np.log(P_xi_spam)
		curprob_ham += np.log(P_xi_ham)
	#last msg
	curprob_ham += np.log(hamNum/dataNum)
	curprob_spam += np.log(spamNum/dataNum)
	if curprob_ham < curprob_spam:
		res.append(1)
	else:
		res.append(0)

	return res

def CompareResult(prediction, testLabel):
	numCorrect = 0
	for i, row in testLabel.iterrows():
		# print (row[0])
		if (prediction[i]==row[0]):
			numCorrect+=1
	return numCorrect / testDataNum
	#print (numCorrect)

# main:
if __name__ == "__main__":

	# command-line input for train-data size
	inputArgSize = len(sys.argv)
	if (inputArgSize > 1):
		train_data_feature += "-" + str(sys.argv[1]) + ".txt"
		train_data_label += "-" + str(sys.argv[1]) + ".txt"
		dataNum = int(sys.argv[1])
		# assumed ham to spam is 1:1 ratio
		hamNum = dataNum/2
		spamNum = dataNum/2
	else:
		train_data_feature += ".txt"
		train_data_label += ".txt"

	# csv data for training
	csvFeatureData = pd.read_csv(train_data_feature, header=None, delimiter = ' ')
	csvLabelData = pd.read_csv(train_data_label, header=None)

	#############################
	######## TRAINING ###########
	#############################
	SpamDict = np.zeros((2501,1),float)
	HamDict = np.zeros((2501,1),float)

	for row in csvFeatureData.iterrows():
		msgNum = int(row[1][0])
		wordIndex = int(row[1][1])
		occurNum = int(row[1][2])
		if (csvLabelData[0][msgNum-1] < 1): # if Ham
			HamDict[wordIndex] += 1
		else: # Spam
			SpamDict[wordIndex] += 1

	#############################
	######### Testing ###########
	#############################
	csvFeatureTestData = pd.read_csv(test_data_feature, header=None, delimiter = ' ')
	csvLabelTestData = pd.read_csv(test_data_label, header=None)
	TestResult = TestBayes(SpamDict, HamDict, csvFeatureTestData)

	#############################
	######## Validating #########
	#############################
	accuracy = CompareResult(TestResult, csvLabelTestData)

	print("Accuracy with training set of %s : %f" % (train_data_feature, accuracy))